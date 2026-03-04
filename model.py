import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba


# ============================================================
# Level-0: 
# ============================================================

class TokenMoEFFN(nn.Module):
    """
    Token-wise MoE FFN: route each agent token to experts.
    x: [B, N, D] -> [B, N, D]
    """
    def __init__(
        self,
        model_dim: int,
        hidden_mult: int = 4,
        n_experts: int = 4,
        dropout: float = 0.1,
        routing: str = "soft",
        temperature: float = 1.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.routing = routing
        self.temperature = temperature

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim * hidden_mult),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim * hidden_mult, model_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

        self.gate = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, n_experts)
        )

        # For debugging/visualization: last routing weights
        self.last_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        logits = self.gate(x) / self.temperature  # [B, N, E]

        if self.routing == "top1":
            idx = torch.argmax(logits, dim=-1, keepdim=True)  # [B, N, 1]
            out = torch.zeros_like(x)
            for e, expert in enumerate(self.experts):
                mask = (idx[..., 0] == e).float().unsqueeze(-1)  # [B, N, 1]
                if mask.any():
                    xe = x * mask
                    ye = expert(xe)
                    out = out + ye

            w = F.one_hot(idx.squeeze(-1), num_classes=self.n_experts).float()
            self.last_gate = w
            return out

        # soft routing
        w = torch.softmax(logits, dim=-1)  # [B, N, E]
        self.last_gate = w.detach()

        outs = [expert(x) for expert in self.experts]  # list of [B, N, D]
        Y = torch.stack(outs, dim=1)                   # [B, E, N, D]
        w_ = w.permute(0, 2, 1).unsqueeze(-1)          # [B, E, N, 1]
        y = (Y * w_).sum(dim=1)                        # [B, N, D]
        return y


class TrajectoryFeatureBuilder(nn.Module):
    """
    Feature construction:
    [pos(2), vel(2), acc(2), onehot_role(3)] -> dim=9
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(positions: torch.Tensor) -> torch.Tensor:
        """
        positions: [B, T, N, 2]
        return:    [B, T, N, 9]
        """
        B, T, N, _ = positions.shape
        device = positions.device

        # Team split (ball + offense + defense), same as original
        N_team = int((N - 1) / 2)

        velocity = torch.zeros_like(positions)
        if T > 1:
            velocity[:, 1:] = positions[:, 1:] - positions[:, :-1]

        acceleration = torch.zeros_like(velocity)
        if T > 1:
            acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]

        category = torch.zeros(B, T, N, 3, device=device)
        category[:, :, 0, 0] = 1
        category[:, :, 1:1 + N_team, 1] = 1
        category[:, :, 1 + N_team:N, 2] = 1

        return torch.cat([positions, velocity, acceleration, category], dim=-1)


# ============================================================
# Level-1: Role-adaptive behavior encoder (Spatial)
# ============================================================

class RoleAdaptiveBehaviorEncoder(nn.Module):
    """
    (Method-level) Role-adaptive Behavior Module:
      - Per-timestep MultiheadAttention over agents
      - FFN replaced by TokenMoEFFN
    Input:  x: [B, T, N, d_in]
    Output: z: [B, T, N, D]
    """
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_agents: int, dropout: float = 0.1):
        super().__init__()
        self.input_embed = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_agents, model_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.ffn = TokenMoEFFN(
            model_dim=model_dim,
            hidden_mult=4,
            n_experts=4,
            dropout=dropout,
            routing="soft",
            temperature=1.0
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, N, input_dim]
        mask: [B, T, N] 
        """
        B, T, N, _ = x.shape
        x_emb = self.input_embed(x)  # [B, T, N, D]

        output = []
        for t in range(T):
            x_t = x_emb[:, t] + self.pos_embed  # [B, N, D]

            x_t = self.norm1(x_t)
            attn_out, _ = self.attention(x_t, x_t, x_t)  # [B, N, D]
            x_t = x_t + attn_out

            x_t = x_t + self.ffn(self.norm2(x_t))  # [B, N, D]
            output.append(x_t)

        return torch.stack(output, dim=1)  # [B, T, N, D]


# ============================================================
# Level-2: Individual dynamics encoder (Temporal)
# ============================================================

class IndividualDynamicsEncoder(nn.Module):
    """
    (Method-level) Individual Dynamics Module:
      - Mamba stack over time (per agent token)
      - Low-rank dynamic bases with gated mixture weights + stable Taylor update
      - Optional FiLM, mix gate
    """
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        conv_kernel: int,
        num_layers: int,
        K_bases: int = 4,
        rank: int = 16,
        max_dt: float = 0.2,
        use_film: bool = True,
        taylor_order: int = 2
    ):
        super().__init__()
        self.D = model_dim
        self.K = K_bases
        self.r = rank
        self.max_dt = max_dt
        self.use_film = use_film
        assert taylor_order in (1, 2)
        self.taylor_order = taylor_order

        self.mamba_stack = nn.ModuleList([
            Mamba(d_model=model_dim, d_state=state_dim, d_conv=conv_kernel)
            for _ in range(num_layers)
        ])

        # Low-rank bases A_k = U_k V_k^T (with scaling)
        self.U = nn.Parameter(torch.randn(self.K, self.D, self.r) * (1.0 / np.sqrt(self.D)))
        self.V = nn.Parameter(torch.randn(self.K, self.r, self.D) * (1.0 / np.sqrt(self.D)))
        self.A_scale = nn.Parameter(torch.ones(self.K) * 0.3)

        # Gate -> (w_k logits, dt_logit)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.K + 1)
        )

        # FiLM
        if self.use_film:
            self.film_gamma = nn.Sequential(nn.Linear(self.D, self.D), nn.Tanh())
            self.film_beta = nn.Sequential(nn.Linear(self.D, self.D), nn.Tanh())

        # Mix intensity gate
        self.mix_gate = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def _run_mamba(self, h: torch.Tensor) -> torch.Tensor:
        # h: [BN, T, D]
        for m in self.mamba_stack:
            h = h + m(h)
        return h

    def _apply_Amix(self, h: torch.Tensor, w: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
        """
        h: [BN, T, D]
        w: [BN, T, K]
        U: [K, D, r]
        V: [K, r, D]
        returns:
          hA  = h @ A_mix
          hA2 = (h @ A_mix) @ A_mix   (if taylor_order==2)
        """
        # hU:  [BN, T, K, r]
        hU = torch.einsum("btd,kdr->btkr", h, U)
        # hUV: [BN, T, K, D]
        hUV = torch.einsum("btkr,krd->btkd", hU, V)

        w_ = w.unsqueeze(-1)       # [BN, T, K, 1]
        hA = (hUV * w_).sum(dim=2) # [BN, T, D]

        if self.taylor_order == 1:
            return hA, None

        hAU = torch.einsum("btd,kdr->btkr", hA, U)
        hAUV = torch.einsum("btkr,krd->btkd", hAU, V)
        hA2 = (hAUV * w_).sum(dim=2)
        return hA, hA2

    def forward(self, x: torch.Tensor, mask=None, use_bts: bool = False) -> torch.Tensor:
        """
        x: [B, T, N, D] -> [B, T, N, D]
        """
        B, T, N, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # [BN, T, D]

        # 1) Mamba encoding
        h = self._run_mamba(h)

        # 2) Gate (w, dt)
        gate_out = self.gate_mlp(h)  # [BN, T, K+1]
        logits, dt_logit = gate_out[..., :self.K], gate_out[..., -1:]
        w = torch.softmax(logits, dim=-1)               # [BN, T, K]
        dt = torch.sigmoid(dt_logit) * self.max_dt      # [BN, T, 1]

        # Stable scaling
        U_eff = self.U * self.A_scale.view(self.K, 1, 1)
        V_eff = self.V

        hA, hA2 = self._apply_Amix(h, w, U_eff, V_eff)

        upd = dt * hA
        if self.taylor_order == 2 and hA2 is not None:
            upd = upd + 0.5 * (dt ** 2) * hA2

        upd_nl = torch.tanh(upd)

        if self.use_film:
            gamma = self.film_gamma(h)
            beta = self.film_beta(h)
            upd_nl = gamma * upd_nl + beta

        g = self.sigmoid(self.mix_gate(h))  # [BN, T, 1]
        h_dyn = h + g * upd_nl

        out = h_dyn.view(B, N, T, D).permute(0, 2, 1, 3)  # [B, T, N, D]
        return out


# ============================================================
# Level-3: Group-level decoder 
# ============================================================

class GroupCoordinationDecoder(nn.Module):
    """
    (Method-level) Group Relational Decoder in your paper.
    IMPORTANT:
      - To NOT change runtime logic, we keep your existing GRU rollout decoder exactly.
    """
    def __init__(self, model_dim: int, z_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(model_dim + z_dim + 2, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, h_obs: torch.Tensor, z: torch.Tensor, pred_len: int, start_pos: torch.Tensor) -> torch.Tensor:
        """
        h_obs:     [B, N, D]
        z:         [B, N, z_dim]
        start_pos: [B, N, 2]
        return:    [B, pred_len, N, 2]
        """
        B, N, _ = h_obs.shape

        current_pos = start_pos
        state = torch.zeros(B * N, self.hidden_dim, device=h_obs.device)

        preds = []
        for _ in range(pred_len):
            step_input = torch.cat([h_obs, z, current_pos], dim=-1)  # [B, N, D+z+2]
            step_input = self.input_proj(step_input).view(B * N, -1)

            state = self.gru_cell(step_input, state)
            state = self.dropout(state)

            delta = self.out_proj(state).view(B, N, 2)
            next_pos = current_pos + delta

            preds.append(next_pos)
            current_pos = next_pos

        return torch.stack(preds, dim=1)


# ============================================================
# CVAE latent intention module (prior/posterior + reparam)
# ============================================================

class CVAELatentIntention(nn.Module):
    """
    Encapsulates:
      - prior p(z|h_obs)
      - posterior q(z|h_obs, h_fut) (training only)
      - reparameterization trick
    """
    def __init__(self, model_dim: int, z_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.z_dim = z_dim

        self.prior_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, z_dim * 2)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, z_dim * 2)
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prior(self, h_obs: torch.Tensor):
        params = self.prior_net(h_obs)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar

    def posterior(self, h_obs: torch.Tensor, h_fut: torch.Tensor):
        h_combined = torch.cat([h_obs, h_fut], dim=-1)
        params = self.posterior_net(h_combined)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar



class H3M(nn.Module):
    """
    H3M-structured UniTraj predictor:
      - FeatureBuilder
      - RoleAdaptiveBehaviorEncoder (Spatial)
      - IndividualDynamicsEncoder (Temporal)
      - CVAE latent module
      - GroupCoordinationDecoder
    """
    def __init__(
        self,
        obs_len: int = 8,
        pred_len: int = 16,
        num_agents: int = 23,
        input_dim: int = 9,
        model_dim: int = 64,
        z_dim: int = 128,
        num_heads: int = 8,
        state_dim: int = 64,
        conv_kernel: int = 4,
        num_mamba_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_agents = num_agents
        self.model_dim = model_dim
        self.z_dim = z_dim

        # Feature construction
        self.feature_builder = TrajectoryFeatureBuilder()

        # Level-1: role-adaptive behavior encoder (spatial)
        self.spatial_encoder = RoleAdaptiveBehaviorEncoder(
            input_dim=input_dim,
            model_dim=model_dim,
            num_heads=num_heads,
            num_agents=num_agents,
            dropout=dropout
        )

        # Level-2: individual dynamics encoder (temporal)
        self.temporal_encoder = IndividualDynamicsEncoder(
            model_dim=model_dim,
            state_dim=state_dim,
            conv_kernel=conv_kernel,
            num_layers=num_mamba_layers
        )

        # CVAE latent intention
        self.cvae = CVAELatentIntention(model_dim=model_dim, z_dim=z_dim)

        # Level-3: group decoder (kept exact GRU logic)
        self.decoder = GroupCoordinationDecoder(model_dim=model_dim, z_dim=z_dim, hidden_dim=128, dropout=dropout)

    # ---------------------------
    # Inference utility
    # ---------------------------
    def inference_best_of_k(self, obs, k=20, ground_truth=None, temperature=1.0, selection_mode="best"):
        """
        EXACT same behavior as your original inference_best_of_k.
        """
        B, T_obs, N, _ = obs.shape
        device = obs.device

        obs_features = self.feature_builder(obs)
        obs_mask = torch.ones(B, T_obs, N, device=device)

        spatial_features = self.spatial_encoder(obs_features, obs_mask)
        temporal_features = self.temporal_encoder(spatial_features, obs_mask)
        h_obs = temporal_features[:, -1]  # [B, N, D]

        predictions_k = []

        z_mu_p, z_logvar_p = self.cvae.prior(h_obs)

        for _ in range(k):
            std = torch.exp(0.5 * z_logvar_p) * temperature
            eps = torch.randn_like(std)
            z = z_mu_p + eps * std

            start_pos = obs[:, -1]
            pred = self.decoder(h_obs, z, self.pred_len, start_pos)
            predictions_k.append(pred)

        predictions_k = torch.stack(predictions_k, dim=0)  # [K, B, T, N, 2]

        if selection_mode == "all":
            return predictions_k

        if selection_mode == "best":
            if ground_truth is None:
                raise ValueError("'best' mode requires ground_truth")
            gt = ground_truth.unsqueeze(0).expand(k, -1, -1, -1, -1)
            ade = torch.norm(predictions_k - gt, dim=-1).mean(dim=(2, 3))  # [K, B]
            best_idx = ade.argmin(dim=0)  # [B]
            best_preds = predictions_k[best_idx, torch.arange(B)]
            return best_preds

        if selection_mode == "diverse":
            mean_pred = predictions_k.mean(dim=0)
            diversity = torch.norm(predictions_k - mean_pred.unsqueeze(0), dim=-1).sum(dim=(2, 3))  # [K, B]
            diverse_idx = diversity.argmax(dim=0)
            diverse_preds = predictions_k[diverse_idx, torch.arange(B)]
            return diverse_preds

        if selection_mode == "confidence":
            velocity = predictions_k[:, :, 1:] - predictions_k[:, :, :-1]
            accel = velocity[:, :, 1:] - velocity[:, :, :-1]
            smoothness = -torch.norm(accel, dim=-1).mean(dim=(2, 3))  # [K, B]
            smooth_idx = smoothness.argmax(dim=0)
            smooth_preds = predictions_k[smooth_idx, torch.arange(B)]
            return smooth_preds

        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, obs, future=None):
        """
        obs:    [B, obs_len, N, 2]
        future: [B, pred_len, N, 2] (training only)
        returns:
          training: (predictions, kl_loss)
          inference: predictions
        """
        B, T_obs, N, _ = obs.shape
        device = obs.device

        obs_features = self.feature_builder(obs)
        obs_mask = torch.ones(B, T_obs, N, device=device)

        # Level-1 (spatial)
        spatial_features = self.spatial_encoder(obs_features, obs_mask)
        # Level-2 (temporal)
        temporal_features = self.temporal_encoder(spatial_features, obs_mask)
        h_obs = temporal_features[:, -1]  # [B, N, D]

        if self.training and future is not None:
            # Encode future (same path as original)
            future_features = self.feature_builder(future)
            future_mask = torch.ones(B, self.pred_len, N, device=device)

            spatial_future = self.spatial_encoder(future_features, future_mask)
            temporal_future = self.temporal_encoder(spatial_future, None, use_bts=False)
            h_fut = temporal_future[:, -1]  # [B, N, D]

            # Posterior q(z|obs,fut)
            z_mu_q, z_logvar_q = self.cvae.posterior(h_obs, h_fut)
            z_q = self.cvae.reparameterize(z_mu_q, z_logvar_q)

            # Prior p(z|obs)
            z_mu_p, z_logvar_p = self.cvae.prior(h_obs)

            # Decode (same GRU rollout)
            start_pos = obs[:, -1]
            predictions = self.decoder(h_obs, z_q, self.pred_len, start_pos)

            # KL loss (exact same formula as your original)
            kl_loss = -0.5 * torch.sum(
                1 + z_logvar_q - z_logvar_p
                - (z_mu_q - z_mu_p).pow(2) / z_logvar_p.exp()
                - z_logvar_q.exp() / z_logvar_p.exp()
            ) / (B * N)

            return predictions, kl_loss

        # Inference: sample from prior
        z_mu_p, z_logvar_p = self.cvae.prior(h_obs)
        z = self.cvae.reparameterize(z_mu_p, z_logvar_p)

        start_pos = obs[:, -1]
        predictions = self.decoder(h_obs, z, self.pred_len, start_pos)
        return predictions


# ============================================================
# Loss
# ============================================================

class H3MLoss(nn.Module):
    """
    MSE + lambda_kl * KL
    """
    def __init__(self, lambda_kl=0.1, lambda_diverse=0.1):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_diverse = lambda_diverse

    def forward(self, pred, target, kl_loss=None):
        pred_loss = F.mse_loss(pred, target)

        total_loss = pred_loss
        if kl_loss is not None:
            total_loss = total_loss + self.lambda_kl * kl_loss

        return total_loss, pred_loss, kl_loss