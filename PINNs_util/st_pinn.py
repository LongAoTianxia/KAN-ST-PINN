import torch
import torch.nn as nn
import math


class ResBlock(nn.Module):
    """残差MLP块，pre-norm结构"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.norm(x)
        h = torch.tanh(self.fc1(h))
        h = self.fc2(h)
        return x + h


class STPinn(nn.Module):
    """ST-PINN v2: 傅里叶特征 + 残差空间编码 + BiLSTM + 交叉注意力"""

    def __init__(
        self,
        d_model: int = 128,
        seq_len: int = 8,
        dt_hist: float = 0.02,
        n_fourier: int = 64,
        spatial_layers: int = 4,
        lstm_layers: int = 2,
        num_heads: int = 4,
        n_attn_layers: int = 2,
        dropout: float = 0.0,
        # 消融实验开关
        use_fourier: bool = True,
        use_lstm: bool = True,
        use_attention: bool = True,
        attn_type: str = 'cross',
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dt_hist = dt_hist
        self.n_fourier = n_fourier
        self.use_fourier = use_fourier and (n_fourier > 0)
        self.use_lstm = use_lstm
        self.use_attention = use_attention
        self.attn_type = attn_type

        self.pi = torch.acos(torch.zeros(1)).item() * 2

        # 预计算时间偏移
        offsets = -torch.arange(seq_len - 1, -1, -1, dtype=torch.float32) * dt_hist
        self.register_buffer("time_offsets", offsets.view(1, seq_len, 1))

        # [A] 傅里叶特征嵌入
        if self.use_fourier:
            self.B_spatial = nn.Parameter(torch.randn(2, n_fourier) * 2.0)
            self.b_spatial = nn.Parameter(torch.zeros(1, n_fourier))
            self.B_temporal = nn.Parameter(torch.randn(1, n_fourier) * 2.0)
            self.b_temporal = nn.Parameter(torch.zeros(1, n_fourier))
            spatial_in_dim = 2 * n_fourier + 2   # sin + cos + raw coords
            temporal_in_dim = 2 * n_fourier + 1   # sin + cos + raw t
        else:
            spatial_in_dim = 2
            temporal_in_dim = 1

        # [B] 空间编码器
        s_layers = [nn.Linear(spatial_in_dim, d_model), nn.Tanh()]
        for _ in range(spatial_layers - 1):
            s_layers.append(ResBlock(d_model))
        self.spatial_encoder = nn.Sequential(*s_layers)

        # [C] 时序编码器
        self.temporal_embed = nn.Linear(temporal_in_dim, d_model)
        # 可学习的时间位置编码
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        if self.use_lstm:
            self.temporal_lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,    # half because bidirectional doubles it
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            # BiLSTM输出映射回d_model维
            self.lstm_proj = nn.Linear(d_model, d_model)
        else:
            # 消融选项：用MLP替代LSTM
            self.temporal_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, d_model),
            )

        # [D] 时空融合
        if self.use_attention:
            if attn_type == 'cross':
                self.attn_layers = nn.ModuleList()
                self.attn_norms = nn.ModuleList()
                self.ffn_layers = nn.ModuleList()
                self.ffn_norms = nn.ModuleList()
                for _ in range(n_attn_layers):
                    self.attn_layers.append(nn.MultiheadAttention(
                        embed_dim=d_model, num_heads=num_heads,
                        dropout=dropout, batch_first=True,
                    ))
                    self.attn_norms.append(nn.LayerNorm(d_model))
                    self.ffn_layers.append(nn.Sequential(
                        nn.Linear(d_model, d_model * 2),
                        nn.GELU(),
                        nn.Linear(d_model * 2, d_model),
                    ))
                    self.ffn_norms.append(nn.LayerNorm(d_model))
            elif attn_type == 'self':
                self.self_attn = nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=num_heads,
                    dropout=dropout, batch_first=True,
                )
                self.attn_norm = nn.LayerNorm(d_model)
            fusion_dim = d_model
        else:
            # 消融选项：去掉注意力，直接拼接
            fusion_dim = 2 * d_model

        # 输出头
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        # Xavier初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # 辅助方法
    def _build_time_window(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B, 1]
        Returns:
            t_seq: [B, seq_len, 1]
        """
        t_seq = t.unsqueeze(1) + self.time_offsets  # [B, seq_len, 1]
        return torch.clamp(t_seq, min=0.0)

    # 前向传播
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: [B, 3]  (x, y, t)
        Returns:
            out: [B, 1]
        """
        B = r.shape[0]
        r_s = r[:, :2]     # [B, 2]
        t   = r[:, 2:3]    # [B, 1]

        # ---- [A] Fourier features (sin + cos + raw) ----
        if self.use_fourier:
            proj_s = r_s @ self.B_spatial + self.b_spatial        # [B, n_fourier]
            phi_s = torch.cat([torch.sin(2 * self.pi * proj_s),
                               torch.cos(2 * self.pi * proj_s),
                               r_s], dim=-1)                      # [B, 2*n_fourier+2]
            # 时间窗口逐步处理
            t_seq = self._build_time_window(t)  # [B, seq_len, 1]
            t_flat = t_seq.reshape(B * self.seq_len, 1)
            proj_t = t_flat @ self.B_temporal + self.b_temporal   # [B*seq, n_fourier]
            phi_t_flat = torch.cat([torch.sin(2 * self.pi * proj_t),
                                    torch.cos(2 * self.pi * proj_t),
                                    t_flat], dim=-1)               # [B*seq, 2*n_fourier+1]
            phi_t = phi_t_flat.reshape(B, self.seq_len, -1)
        else:
            phi_s = r_s
            t_seq = self._build_time_window(t)
            t_flat = t_seq.reshape(B * self.seq_len, 1)
            phi_t_flat = t_flat
            phi_t = t_seq  # [B, seq_len, 1]

        # ---- [B] Spatial Encoder ----
        s_feat = self.spatial_encoder(phi_s)  # [B, d_model]

        # ---- [C] Temporal Encoder ----
        phi_t_2d = phi_t.reshape(B * self.seq_len, -1)
        X_t_flat = self.temporal_embed(phi_t_2d)                      # [B*seq_len, d_model]
        X_t = X_t_flat.reshape(B, self.seq_len, self.d_model)        # [B, seq_len, d_model]
        X_t = X_t + self.temporal_pos_emb                              # add position embeddings

        if self.use_lstm:
            H, _ = self.temporal_lstm(X_t)       # [B, seq_len, d_model] (BiLSTM)
            H = self.lstm_proj(H)                # project back to d_model
            t_tokens = H
            t_last = H[:, -1, :]
        else:
            t_out_flat = self.temporal_mlp(X_t_flat)
            t_out = t_out_flat.reshape(B, self.seq_len, self.d_model)
            t_tokens = t_out
            t_last = t_out[:, -1, :]

        # ---- [D] Fusion ----
        if self.use_attention:
            s_query = s_feat.unsqueeze(1)  # [B, 1, d_model]

            if self.attn_type == 'cross':
                q = s_query
                for attn_layer, attn_norm, ffn, ffn_norm in zip(
                    self.attn_layers, self.attn_norms,
                    self.ffn_layers, self.ffn_norms
                ):
                    attn_out, _ = attn_layer(q, t_tokens, t_tokens)
                    q = attn_norm(q + attn_out)
                    ffn_out = ffn(q)
                    q = ffn_norm(q + ffn_out)
                fused = q.squeeze(1)  # [B, d_model]
            elif self.attn_type == 'self':
                tokens = torch.cat([s_query, t_tokens], dim=1)
                attn_out, _ = self.self_attn(tokens, tokens, tokens)
                tokens = self.attn_norm(tokens + attn_out)
                fused = tokens[:, 0, :]
        else:
            fused = torch.cat([s_feat, t_last], dim=-1)  # [B, 2*d_model]

        # ---- Output ----
        out = self.head(fused)  # [B, 1]
        return out
