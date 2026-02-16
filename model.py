import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import numpy as np

class TokenMoEFFN(nn.Module):
    """
    Route to multiple expert MLPs token by token (per agent), then weighted sum by gating weights
    """
    def __init__(self, model_dim, hidden_mult=4, n_experts=4, dropout=0.1,
                 routing='soft', temperature=1.0):
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
                nn.Dropout(dropout)
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, n_experts)
        )

        self.last_gate = None

    def forward(self, x):
        """
        x: [B, N, D]
        return: [B, N, D]
        """
        B, N, D = x.shape
        logits = self.gate(x) / self.temperature   # [B, N, E]

        if self.routing == 'top1':  
            idx = torch.argmax(logits, dim=-1, keepdim=True)   # [B,N,1]
            out = torch.zeros_like(x)
            for e, expert in enumerate(self.experts):
                mask = (idx[..., 0] == e).float().unsqueeze(-1)  # [B,N,1]
                if mask.any():
                    xe = x * mask
                    ye = expert(xe)
                    out = out + ye
            # record（one-hot）
            w = F.one_hot(idx.squeeze(-1), num_classes=self.n_experts).float()
            self.last_gate = w
            return out

        else:  # 'soft'：
            w = torch.softmax(logits, dim=-1)  # [B,N,E]
            self.last_gate = w.detach()

            # Calculate all experts and weight the sum）
            outs = []
            outs = [expert(x) for expert in self.experts]   # [B,N,D]
            Y = torch.stack(outs, dim=1)                    # [B,E,N,D]
            w_ = w.permute(0,2,1).unsqueeze(-1)             # [B,E,N,1]
            y = (Y * w_).sum(dim=1)                         # [B,N,D]

            return y



class MinimalSpatialEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_agents, dropout=0.1):
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
        # self.ffn = nn.Sequential(
        #     nn.Linear(model_dim, model_dim * 4),
        #     nn.LeakyReLU(),
        #     nn.Linear(model_dim * 4, model_dim),
        #     nn.Dropout(dropout)
        # )
        self.ffn = TokenMoEFFN(model_dim, hidden_mult=4, n_experts=4, dropout=dropout,
                       routing='soft', temperature=1.0)
        
    def forward(self, x, mask):
        B, T, N, _ = x.shape
        x_emb = self.input_embed(x)
        
        output = []
        for t in range(T):
            x_t = x_emb[:, t] + self.pos_embed
            
            x_t = self.norm1(x_t)
            attn_out, _ = self.attention(x_t, x_t, x_t)
            x_t = x_t + attn_out
            x_t = x_t + self.ffn(self.norm2(x_t))
            
            output.append(x_t)
        
        output = torch.stack(output, dim=1)
        return output




class TemporalMambaEncoder(nn.Module):
    """
    Mamba residual heap + dynamic low-rank dynamic basis combination + local time stretching + second-order stable update + FiLM
    """
    def __init__(self, model_dim, state_dim, conv_kernel, num_layers,
                 moe_num_experts=0, moe_top_k=0, moe_after_each=False, moe_dropout=0.0,
                 K_bases=4,    # Number of low-rank dynamic bases (recommended 4~8)
                 rank=16,      # Low rank size of each basis (recommended 8~32)
                 max_dt=0.2,  # Maximum step size for time stretching
                 use_film=True,
                 taylor_order=2  
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

        # 2) Low-rank dynamic basis (A_k = U_k V_k^T) with learnable scaling to ensure stability
    
        self.U = nn.Parameter(torch.randn(self.K, self.D, self.r) * (1.0 / np.sqrt(self.D)))
        self.V = nn.Parameter(torch.randn(self.K, self.r, self.D) * (1.0 / np.sqrt(self.D)))
        self.A_scale = nn.Parameter(torch.ones(self.K) * 0.3)  

        # 3) Gating: Obtain (w_k, dt) from the hidden state
        # - w_k: Mixed weights of K basis points (softmax)
        # - dt: Local step size of [0, max_dt] (sigmoid * max_dt)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.K + 1)  # The first K are logits, and the last one is dt_logit.
        )

        # 4) FiLM (optional): Scale/bias the update per token.
        if self.use_film:
            self.film_gamma = nn.Sequential(nn.Linear(self.D, self.D), nn.Tanh())
            self.film_beta  = nn.Sequential(nn.Linear(self.D, self.D), nn.Tanh())

        # 5) Synthesis intensity gate controls "how much kinetics to apply" to prevent excessive strength in the early stages.
        self.mix_gate = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def _run_mamba(self, h):
        # h: [B*N, T, D]
        for m in self.mamba_stack:
            h = h + m(h)
        return h

    # ---------- computing A_mix*h and (A_mix^2)*h ----------
    def _apply_Amix(self, h, w, U, V):
        """
        h: [BN, T, D]
        w: [BN, T, K]  
        U: [K, D, r]
        V: [K, r, D]
        return:
          hA  = h @ A_mix
          hA2 = (h @ A_mix) @ A_mix
        """
        BN, T, D = h.shape
        # hU:  [BN,T,K,r]   (h @ U_k)
        hU  = torch.einsum('btd,kdr->btkr', h, U)
        # hUV: [BN,T,K,D]   (h @ U_k) @ V_k^T
        hUV = torch.einsum('btkr,krd->btkd', hU, V)
        # sum_k w_k * hUV_k
        w_  = w.unsqueeze(-1)            # [BN,T,K,1]
        hA  = (hUV * w_).sum(dim=2)      # [BN,T,D]

        if self.taylor_order == 1:
            return hA, None

        # hA2 = (hA) @ A_mix
        hAU  = torch.einsum('btd,kdr->btkr', hA, U)     # [BN,T,K,r]
        hAUV = torch.einsum('btkr,krd->btkd', hAU, V)   # [BN,T,K,D]
        hA2  = (hAUV * w_).sum(dim=2)                   # [BN,T,D]
        return hA, hA2


    def forward(self, x, mask=None, use_bts=False):
        """
        x: [B, T, N, D] -> [B, T, N, D]
        """
        B, T, N, D = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # [BN,T,D]

        # 1) Mamba 编码
        h = self._run_mamba(h)                          # [BN,T,D]

        # 2) 门控出 (w_k, dt)，再做稳定的二阶更新
        gate_out = self.gate_mlp(h)                     # [BN,T,K+1]
        logits, dt_logit = gate_out[..., :self.K], gate_out[..., -1:]
        w = torch.softmax(logits, dim=-1)               # [BN,T,K]
        dt = torch.sigmoid(dt_logit) * self.max_dt      # [BN,T,1]

        # 稳定缩放各基（不改写参数本体）
        U_eff = self.U * self.A_scale.view(self.K, 1, 1)  # [K,D,r]
        V_eff = self.V                                    # [K,r,D]
        hA, hA2 = self._apply_Amix(h, w, U_eff, V_eff)


        # 一阶或二阶泰勒：h + dt*hA + 0.5*dt^2*hA2
        upd = dt * hA
        if self.taylor_order == 2 and hA2 is not None:
            upd = upd + 0.5 * (dt ** 2) * hA2

        # 非线性稳定：tanh + 逐 token 门控
        upd_nl = torch.tanh(upd)

        # FiLM 调制（可选）
        if self.use_film:
            gamma = self.film_gamma(h)  # [BN,T,D]
            beta  = self.film_beta(h)   # [BN,T,D]
            upd_nl = gamma * upd_nl + beta

        # 混合强度门
        g = self.sigmoid(self.mix_gate(h))  # [BN,T,1]
        h_dyn = h + g * upd_nl              # [BN,T,D]

        # 回到 [B,T,N,D]
        out = h_dyn.view(B, N, T, D).permute(0, 2, 1, 3)
        return out
    
class RecurrentDecoder(nn.Module):
    """
    GRU 递推式解码器：每一步基于上一时刻预测逐帧生成未来轨迹
    """
    def __init__(self, model_dim, z_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 将观测特征 h_obs 与 z 编码后输入 GRU
        self.input_proj = nn.Linear(model_dim + z_dim + 2, hidden_dim)  
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, 2)  # 输出 delta 位置

    def forward(self, h_obs, z, pred_len, start_pos):
        """
        h_obs: [B, N, D]
        z: [B, N, z_dim]
        start_pos: [B, N, 2] 当前最后观测帧的位置
        return: pred [B, pred_len, N, 2]
        """
        B, N, _ = h_obs.shape

        # 初始输入：concat(h_obs, z, current_pos)
        current_pos = start_pos  # 当前时间步位置
        state = torch.zeros(B * N, self.hidden_dim, device=h_obs.device)

        preds = []
        for t in range(pred_len):
            # 将当前时间步的位置作为输入的一部分
            step_input = torch.cat([h_obs, z, current_pos], dim=-1)  # [B, N, D+z_dim+2]
            step_input = self.input_proj(step_input).view(B * N, -1)

            # GRU 更新
            state = self.gru_cell(step_input, state)
            state = self.dropout(state)

            # 预测当前位置的位移
            delta = self.out_proj(state).view(B, N, 2)
            next_pos = current_pos + delta  # 位移叠加得到下一时刻位置

            preds.append(next_pos)
            current_pos = next_pos

        # 堆叠所有时间步
        return torch.stack(preds, dim=1)  # [B, pred_len, N, 2]


class UniTrajPredictor(nn.Module):
    """完整的UniTraj预测模型"""
    def __init__(self, obs_len=8, pred_len=16, num_agents=23,
                 input_dim=9, model_dim=64, z_dim=128,
                 num_heads=8, state_dim=64, conv_kernel=4,
                 num_mamba_layers=4, dropout=0.1):
        super().__init__()
        
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_agents = num_agents
        self.model_dim = model_dim
        self.z_dim = z_dim

        self.spatial_encoder = MinimalSpatialEncoder(  # 或 MinimalSpatialEncoder
        input_dim, model_dim, num_heads, num_agents, dropout
        )
        
        self.temporal_encoder = TemporalMambaEncoder(
            model_dim, state_dim, conv_kernel, num_mamba_layers
        )
        
        # CVAE组件
        # Prior network p(z|x_obs)
        self.prior_net = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, z_dim * 2)  # mean and log_var
        )
        
        # Posterior network q(z|x_obs, x_fut) - 训练时使用
        self.posterior_net = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, z_dim * 2)  # mean and log_var
        )
        
        # 解码器
        # self.decoder = nn.Sequential(
        #     nn.Linear(model_dim + z_dim, model_dim * 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(model_dim * 2, pred_len * 2)
        # )
        # 使用 GRU 递推解码器
        self.decoder = RecurrentDecoder(model_dim, z_dim, hidden_dim=128, dropout=dropout)



   

    def inference_best_of_k(self, obs,k=20, ground_truth=None, temperature=1.0, selection_mode='best'):
        """
        生成 K 条轨迹，支持多种选择策略
        
        Args:
            obs: [B, T_obs, N, 2] 观测轨迹
            k: 生成轨迹数量
            ground_truth: [B, T_pred, N, 2] 真实未来轨迹（可选）
            temperature: 采样温度，控制多样性
            selection_mode: 选择模式
                - 'best': 选择ADE最小的（需要ground_truth）
                - 'all': 返回所有K条轨迹
                - 'diverse': 选择最多样化的K条（用于测试展示）
                - 'confidence': 基于模型置信度选择（可扩展）
        
        Returns:
            如果 selection_mode='all': [K, B, T_pred, N, 2]
            否则: [B, T_pred, N, 2]
        """
        B, T_obs, N, _ = obs.shape
        device = obs.device
        
        # Step 1. 计算特征
        obs_features = self.compute_features(obs)
        obs_mask = torch.ones(B, T_obs, N, device=device)
        spatial_features = self.spatial_encoder(obs_features, obs_mask)
        temporal_features = self.temporal_encoder(spatial_features, obs_mask)
        h_obs = temporal_features[:, -1]
        
        # Step 2. 生成 K 条轨迹
        predictions_k = []
        prior_params = self.prior_net(h_obs)  # [B, N, 2*z_dim]
        z_mu_p, z_logvar_p = torch.chunk(prior_params, 2, dim=-1)
        
        for _ in range(k):
            std = torch.exp(0.5 * z_logvar_p) * temperature
            eps = torch.randn_like(std)
            z = z_mu_p + eps * std
            
            start_pos = obs[:, -1]  # [B, N, 2]
            pred = self.decoder(h_obs, z, self.pred_len, start_pos)  # [B, T_pred, N, 2]
            predictions_k.append(pred)
        
        predictions_k = torch.stack(predictions_k, dim=0)  # [K, B, T, N, 2]
        
        # Step 3. 根据模式选择轨迹
        if selection_mode == 'all':
            return predictions_k
        
        elif selection_mode == 'best':
            if ground_truth is None:
                raise ValueError("'best' mode requires ground_truth")
            
            # 计算每条轨迹的 ADE
            gt = ground_truth.unsqueeze(0).expand(k, -1, -1, -1, -1)
            ade = torch.norm(predictions_k - gt, dim=-1).mean(dim=(2, 3))  # [K, B]
            best_idx = ade.argmin(dim=0)  # [B]
            
            # 选择每个batch的最优轨迹
            best_preds = predictions_k[best_idx, torch.arange(B)]
            return best_preds  # [B, T, N, 2]
        
        elif selection_mode == 'diverse':
            # 选择最多样化的轨迹（用于可视化）
            # 简单策略：选择与均值轨迹偏差最大的
            mean_pred = predictions_k.mean(dim=0)  # [B, T, N, 2]
            diversity = torch.norm(predictions_k - mean_pred.unsqueeze(0), dim=-1).sum(dim=(2, 3))  # [K, B]
            diverse_idx = diversity.argmax(dim=0)  # [B]
            diverse_preds = predictions_k[diverse_idx, torch.arange(B)]
            return diverse_preds
        
        elif selection_mode == 'confidence':
            # 基于模型置信度选择（这里用轨迹平滑度作为代理）
            # 计算轨迹的平滑度（速度变化越小越平滑）
            velocity = predictions_k[:, :, 1:] - predictions_k[:, :, :-1]  # [K, B, T-1, N, 2]
            accel = velocity[:, :, 1:] - velocity[:, :, :-1]  # [K, B, T-2, N, 2]
            smoothness = -torch.norm(accel, dim=-1).mean(dim=(2, 3))  # [K, B], 越大越平滑
            smooth_idx = smoothness.argmax(dim=0)  # [B]
            smooth_preds = predictions_k[smooth_idx, torch.arange(B)]
            return smooth_preds
        
        else:
            raise ValueError(f"Unknown selection_mode: {selection_mode}")

    
        
    def compute_features(self, positions):
        """计算输入特征"""
        B, T, N, _ = positions.shape
        N_team = int((N-1)/2) # - ball
        device = positions.device
        
        # 速度
        velocity = torch.zeros_like(positions)
        if T > 1:
            velocity[:, 1:] = positions[:, 1:] - positions[:, :-1]
        
        # 加速度
        acceleration = torch.zeros_like(velocity)
        if T > 1:
            acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        
        # 类别编码
        category = torch.zeros(B, T, N, 3, device=device)
        category[:, :, 0, 0] = 1  # 球
        category[:, :, 1:1+N_team, 1] = 1  
        category[:, :, 1+N_team:N, 2] = 1  
        
        return torch.cat([positions, velocity, acceleration, category], dim=-1)
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, obs, future=None):
        """
        obs: [B, obs_len, N, 2]
        future: [B, pred_len, N, 2] (训练时提供)
        """
        B, T_obs, N, _ = obs.shape
        device = obs.device
        
        # 计算特征
        obs_features = self.compute_features(obs)  # [B, T_obs, N, 9]
        # print("obs_features.shape:", obs_features.shape)
        
        # 创建mask（预测任务中观察部分全为1）
        obs_mask = torch.ones(B, T_obs, N, device=device)


        
        # 空间编码
        spatial_features = self.spatial_encoder(obs_features, obs_mask)
        
        # 时序编码（使用BTS）
        temporal_features = self.temporal_encoder(spatial_features,  obs_mask)
        
        # 聚合时序特征（使用最后时刻）
        h_obs = temporal_features[:, -1]  # [B, N, model_dim]
        
        if self.training and future is not None:
            # 训练模式：使用后验分布
            # 编码未来轨迹
            future_features = self.compute_features(future)
            future_mask = torch.ones(B, self.pred_len, N, device=device)
            
            spatial_future = self.spatial_encoder(future_features, future_mask)
            temporal_future = self.temporal_encoder(spatial_future, None, use_bts=False)
            h_fut = temporal_future[:, -1]  # [B, N, model_dim]
            
            # 后验分布 q(z|x_obs, x_fut)
            h_combined = torch.cat([h_obs, h_fut], dim=-1)
            posterior_params = self.posterior_net(h_combined)
            z_mu_q, z_logvar_q = torch.chunk(posterior_params, 2, dim=-1)
            z_q = self.reparameterize(z_mu_q, z_logvar_q)
            
            # 先验分布 p(z|x_obs)
            prior_params = self.prior_net(h_obs)
            z_mu_p, z_logvar_p = torch.chunk(prior_params, 2, dim=-1)
            
            # 解码
            # decoder_input = torch.cat([h_obs, z_q], dim=-1)
            # predictions = self.decoder(decoder_input)
            # predictions = predictions.reshape(B, N, self.pred_len, 2)
            # predictions = predictions.permute(0, 2, 1, 3)  # [B, pred_len, N, 2]

            # GRU 递推解码
            start_pos = obs[:, -1]  # 最后一个观测帧的位置 [B, N, 2]
            predictions = self.decoder(h_obs, z_q, self.pred_len, start_pos)  # [B, T_pred, N, 2]


            # 计算KL散度
            kl_loss = -0.5 * torch.sum(
                1 + z_logvar_q - z_logvar_p - 
                (z_mu_q - z_mu_p).pow(2) / z_logvar_p.exp() - 
                z_logvar_q.exp() / z_logvar_p.exp()
            ) / (B * N)
            
            return predictions, kl_loss
            
        else:
            # 推理模式：从先验分布采样
            prior_params = self.prior_net(h_obs)
            z_mu_p, z_logvar_p = torch.chunk(prior_params, 2, dim=-1)
            z = self.reparameterize(z_mu_p, z_logvar_p)
            
            start_pos = obs[:, -1]
            predictions = self.decoder(h_obs, z, self.pred_len, start_pos)

            
            return predictions


class UniTrajLoss(nn.Module):
    """UniTraj的损失函数"""
    def __init__(self, lambda_kl=0.1, lambda_diverse=0.1):
        super().__init__()
        self.lambda_kl = lambda_kl
        self.lambda_diverse = lambda_diverse
        
    def forward(self, pred, target, kl_loss=None):
        # 预测损失
        pred_loss = F.mse_loss(pred, target)

        # 总损失
        total_loss = pred_loss
        if kl_loss is not None:
            total_loss = total_loss + self.lambda_kl * kl_loss
            
        return total_loss, pred_loss, kl_loss
    


# # MoE 正则损失
# class UniTrajLoss(nn.Module):
#     def __init__(self, lambda_kl=0.1, lambda_diverse=0.1,
#                  lambda_moe_lb=1e-3, lambda_moe_sp=1e-4):
#         super().__init__()
#         self.lambda_kl = lambda_kl
#         self.lambda_diverse = lambda_diverse
#         self.lambda_moe_lb = lambda_moe_lb
#         self.lambda_moe_sp = lambda_moe_sp

#     def _collect_moe_regs(self, model: nn.Module, device):
#         lb_total, sp_total = None, None
#         for m in model.modules():
#             if hasattr(m, 'loss_load_balance') and hasattr(m, 'loss_sparsity'):
#                 lb = m.loss_load_balance()
#                 sp = m.loss_sparsity()
#                 if isinstance(lb, float): lb = None
#                 if isinstance(sp, float): sp = None
#                 if lb is not None: lb_total = lb if lb_total is None else lb_total + lb
#                 if sp is not None: sp_total = sp if sp_total is None else sp_total + sp
#         if lb_total is None: lb_total = torch.tensor(0.0, device=device)
#         if sp_total is None: sp_total = torch.tensor(0.0, device=device)
#         return lb_total, sp_total
        
#     def forward(self, pred, target, kl_loss=None, model: nn.Module=None):
#         pred_loss = F.mse_loss(pred, target)
#         total_loss = pred_loss
#         if kl_loss is not None:
#             total_loss = total_loss + self.lambda_kl * kl_loss

#         if model is not None:
#             moe_lb_val, moe_sp_val = self._collect_moe_regs(model, pred.device)
#             total_loss = total_loss + self.lambda_moe_lb * moe_lb_val + self.lambda_moe_sp * moe_sp_val

#         return total_loss, pred_loss, kl_loss
