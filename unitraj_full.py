import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import numpy as np


class MinimalSpatialEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_agents, dropout=0.1):
        super().__init__()
        
        self.input_embed = nn.Linear(input_dim, model_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_agents, model_dim))
        
        # 直接在agents间做attention，不需要额外token
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 其他层保持不变
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout)
        )
        
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
    def __init__(self, model_dim, state_dim, conv_kernel, num_layers):
        super().__init__()
        # 删除这行：self.bts = BidirectionalTemporalScaled(model_dim)
        
        # 保持双向Mamba
        self.forward_mamba = nn.ModuleList([
            Mamba(d_model=model_dim, d_state=state_dim, d_conv=conv_kernel)
            for _ in range(num_layers)
        ])
        
        self.backward_mamba = nn.ModuleList([
            Mamba(d_model=model_dim, d_state=state_dim, d_conv=conv_kernel)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None, use_bts=False):  # 保留接口但不使用
        B, T, N, D = x.shape
        x_reshape = x.permute(0, 2, 1, 3).reshape(B*N, T, D)
        
        # 删除所有BTS相关代码，直接处理
        # 前向Mamba
        x_forward = x_reshape
        for mamba_layer in self.forward_mamba:
            x_forward = x_forward + mamba_layer(x_forward)
        
        # 后向Mamba  
        x_backward = torch.flip(x_reshape, dims=[1])
        for mamba_layer in self.backward_mamba:
            x_backward = x_backward + mamba_layer(x_backward)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # 合并
        x_combined = (x_forward + x_backward) / 2
        x_combined = x_combined.reshape(B, N, T, D).permute(0, 2, 1, 3)
        
        return x_combined
    
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
        
        # 空间编码器（带GSM）
        # self.spatial_encoder = SpatialEncoder(
        #     input_dim, model_dim, num_heads, num_agents, dropout
        # )
        self.spatial_encoder = MinimalSpatialEncoder(  # 或 MinimalSpatialEncoder
        input_dim, model_dim, num_heads, num_agents, dropout
        )
        
        # 时序编码器（带BTS的Mamba）
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
        self.decoder = nn.Sequential(
            nn.Linear(model_dim + z_dim, model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, pred_len * 2)
        )
        
        
    def compute_features(self, positions):
        """计算输入特征"""
        B, T, N, _ = positions.shape
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
        category[:, :, 1:12, 1] = 1  # 进攻
        category[:, :, 12:, 2] = 1  # 防守
        
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
        temporal_features = self.temporal_encoder(spatial_features, None)
        
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
            decoder_input = torch.cat([h_obs, z_q], dim=-1)
            predictions = self.decoder(decoder_input)
            predictions = predictions.reshape(B, N, self.pred_len, 2)
            predictions = predictions.permute(0, 2, 1, 3)  # [B, pred_len, N, 2]



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
            
            decoder_input = torch.cat([h_obs, z], dim=-1)
            predictions = self.decoder(decoder_input)
            predictions = predictions.reshape(B, N, self.pred_len, 2)
            predictions = predictions.permute(0, 2, 1, 3)
            
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