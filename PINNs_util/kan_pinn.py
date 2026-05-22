"""
KAN-PINN 二维声波方程求解

两种模型:
1. KANPinn (纯KAN):  (x,y,t) → 傅里叶特征 → KAN层 → p
2. KANSTPinn (KAN+LSTM+Attention): 空间KAN编码 + BiLSTM + 交叉注意力

参考:
    Wang et al., CMAME 2025 (KINN)
    Liu et al., 2024 (KAN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLayer(nn.Module):
    """单层KAN: d_in → d_out, B样条可学习激活 + 残差线性通路"""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: list = [-1.0, 1.0],
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.grid_size = grid_size
        self.spline_order = spline_order

        # B样条均匀节点网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        grid = grid.unsqueeze(0).expand(d_in, -1)  # [d_in, G+2k+1]
        self.register_buffer("grid", grid)

        # 可学习的样条系数
        self.spline_weight = nn.Parameter(
            torch.randn(d_out, d_in, grid_size + spline_order) * scale_noise * scale_spline
        )

        # 残差线性通路
        self.base_weight = nn.Parameter(
            torch.randn(d_out, d_in) * (scale_base / math.sqrt(d_in))
        )

        # 输出缩放因子
        self.spline_scaler = nn.Parameter(torch.ones(d_out, d_in) * scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor递推算B样条基函数"""
        x = x.unsqueeze(-1)  # [batch, d_in, 1]
        grid = self.grid      # [d_in, G+2k+1]

        # 0阶基函数（指示函数）
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()  # [batch, d_in, G+2k]

        # 递归求解k阶B样条基函数
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :-(k + 1)]
            left_den = grid[:, k:-1] - grid[:, :-(k + 1)]
            right_num = grid[:, k + 1:] - x
            right_den = grid[:, k + 1:] - grid[:, 1:(-k)]

            left = left_num / (left_den + 1e-8) * bases[:, :, :-1]
            right = right_num / (right_den + 1e-8) * bases[:, :, 1:]
            bases = left + right

        return bases  # [batch, d_in, G+k]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_in]
        Returns:
            y: [batch, d_out]
        """
        # 残差线性通路
        base_output = F.silu(x) @ self.base_weight.T  # [batch, d_out]

        # 样条路径：B样条基×系数
        spline_basis = self.b_splines(x)  # [batch, d_in, G+k]

        # einsum收缩样条基和系数
        spline_output = torch.einsum('bik,oik->boi', spline_basis, self.spline_weight)

        # 逐边缩放后对输入维求和
        spline_output = (spline_output * self.spline_scaler.unsqueeze(0)).sum(dim=-1)  # [batch, d_out]

        return base_output + spline_output


class KANPinn(nn.Module):
    """纯KAN-PINN: 傅里叶特征 → KAN层 → 声压"""

    def __init__(
        self,
        d_hidden: int = 64,
        n_layers: int = 3,
        n_fourier: int = 32,
        grid_size: int = 5,
        spline_order: int = 3,
        use_fourier: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_fourier = n_fourier
        self.use_fourier = use_fourier and (n_fourier > 0)
        self.use_layernorm = use_layernorm

        self.pi = torch.acos(torch.zeros(1)).item() * 2

        # 傅里叶特征嵌入
        if self.use_fourier:
            # 可学习频率矩阵
            self.B_spatial = nn.Parameter(torch.randn(2, n_fourier) * 2.0)
            self.b_spatial = nn.Parameter(torch.zeros(1, n_fourier))
            self.B_temporal = nn.Parameter(torch.randn(1, n_fourier) * 2.0)
            self.b_temporal = nn.Parameter(torch.zeros(1, n_fourier))
            # sin+cos+原始坐标
            input_dim = 4 * n_fourier + 3
        else:
            input_dim = 3  # raw (x, y, t)

        # KAN网络
        self.kan_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 输入层
        self.kan_layers.append(KANLayer(
            d_in=input_dim, d_out=d_hidden,
            grid_size=grid_size, spline_order=spline_order,
        ))
        if use_layernorm:
            self.norms.append(nn.LayerNorm(d_hidden))

        # 隐藏层
        for _ in range(n_layers - 1):
            self.kan_layers.append(KANLayer(
                d_in=d_hidden, d_out=d_hidden,
                grid_size=grid_size, spline_order=spline_order,
            ))
            if use_layernorm:
                self.norms.append(nn.LayerNorm(d_hidden))

        # 输出层
        self.kan_output = KANLayer(
            d_in=d_hidden, d_out=1,
            grid_size=grid_size, spline_order=spline_order,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: [B, 3]  (x, y, t)
        Returns:
            out: [B, 1]
        """
        r_s = r[:, :2]     # [B, 2]  spatial
        t   = r[:, 2:3]    # [B, 1]  temporal

        # 傅里叶特征
        if self.use_fourier:
            proj_s = r_s @ self.B_spatial + self.b_spatial        # [B, n_fourier]
            proj_t = t @ self.B_temporal + self.b_temporal         # [B, n_fourier]
            phi = torch.cat([
                torch.sin(2 * self.pi * proj_s),
                torch.cos(2 * self.pi * proj_s),
                torch.sin(2 * self.pi * proj_t),
                torch.cos(2 * self.pi * proj_t),
                r_s, t,
            ], dim=-1)  # [B, 4*n_fourier + 3]
        else:
            phi = r  # [B, 3]

        # KAN层
        h = phi
        for i, kan in enumerate(self.kan_layers):
            h = kan(h)
            if self.use_layernorm:
                h = self.norms[i](h)

        # 输出
        out = self.kan_output(h)  # [B, 1]
        return out

    def count_parameters(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        spline_params = 0
        base_params = 0
        fourier_params = 0
        other_params = 0
        for name, p in self.named_parameters():
            n = p.numel()
            if 'spline' in name:
                spline_params += n
            elif 'base_weight' in name:
                base_params += n
            elif 'B_spatial' in name or 'B_temporal' in name or 'b_spatial' in name or 'b_temporal' in name:
                fourier_params += n
            else:
                other_params += n
        return {
            'total': total,
            'spline': spline_params,
            'base_linear': base_params,
            'fourier': fourier_params,
            'other': other_params,
        }


# =======
# KAN-ST-PINN: KAN + BiLSTM + Cross-Attention
# =======

class KANSTPinn(nn.Module):
    """
    KAN-ST-PINN: 空间KAN编码 + BiLSTM时序 + 交叉注意力融合
    
    temporal_mode:
        - "bilstm_attention": KAN + BiLSTM + cross-attention
        - "bilstm_no_attention": KAN + BiLSTM, last pooling + fusion projection
        - "attention_no_bilstm": KAN + temporal token MLP + cross-attention

    """

    def __init__(
        self,
        d_model: int = 128,
        seq_len: int = 8,
        dt_hist: float = 0.02,
        n_fourier: int = 32,
        spatial_kan_layers: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
        lstm_layers: int = 2,
        num_heads: int = 4,
        n_attn_layers: int = 2,
        dropout: float = 0.0,
        temporal_mode: str = "bilstm_attention",
    ):
        super().__init__()

        allowed_modes = {
            "bilstm_attention",
            "bilstm_no_attention",
            "attention_no_bilstm",
        }
        if temporal_mode not in allowed_modes:
            raise ValueError(
                f"Unknown temporal_mode={temporal_mode!r}. "
                f"Expected one of {sorted(allowed_modes)}."
            )
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.dt_hist = dt_hist
        self.n_fourier = n_fourier
        self.temporal_mode = temporal_mode
        self.use_lstm = temporal_mode in {"bilstm_attention", "bilstm_no_attention"}
        self.use_attention = temporal_mode in {"bilstm_attention", "attention_no_bilstm"}


        self.pi = torch.acos(torch.zeros(1)).item() * 2

        # 时间偏移缓存
        offsets = -torch.arange(seq_len - 1, -1, -1, dtype=torch.float32) * dt_hist
        self.register_buffer("time_offsets", offsets.view(1, seq_len, 1))

        # [A] 傅里叶特征
        self.B_spatial = nn.Parameter(torch.randn(2, n_fourier) * 2.0)
        self.b_spatial = nn.Parameter(torch.zeros(1, n_fourier))
        self.B_temporal = nn.Parameter(torch.randn(1, n_fourier) * 2.0)
        self.b_temporal = nn.Parameter(torch.zeros(1, n_fourier))
        
        spatial_in_dim = 2 * n_fourier + 2   # sin + cos + raw (x,y)
        temporal_in_dim = 2 * n_fourier + 1   # sin + cos + raw t

        # [B] 空间KAN编码器
        self.spatial_kan_layers = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()

        # 第一层: fourier_dim → d_model
        self.spatial_kan_layers.append(KANLayer(
            d_in=spatial_in_dim, d_out=d_model,
            grid_size=grid_size, spline_order=spline_order,
        ))
        self.spatial_norms.append(nn.LayerNorm(d_model))

        # 后续KAN层
        for _ in range(spatial_kan_layers - 1):
            self.spatial_kan_layers.append(KANLayer(
                d_in=d_model, d_out=d_model,
                grid_size=grid_size, spline_order=spline_order,
            ))
            self.spatial_norms.append(nn.LayerNorm(d_model))

        # [C] 时序编码器 (BiLSTM)
        self.temporal_embed = nn.Linear(temporal_in_dim, d_model)
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        if self.use_lstm:
            self.temporal_lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            self.lstm_proj = nn.Linear(d_model, d_model)
        else:
            # Attention-only ablation: keep temporal tokens, 用轻量级MLP替代BiLSTM 
            self.temporal_token_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )


        # [D] 交叉注意力
        if self.use_attention:
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
        else:
            # 非 attention 的时间聚合与融合层
            self.temporal_pool = "last"  # 用历史窗口预测当前时刻
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        # [E] KAN输出头
        self.head_kan = KANLayer(
            d_in=d_model, d_out=d_model // 2,
            grid_size=grid_size, spline_order=spline_order,
        )
        self.head_norm = nn.LayerNorm(d_model // 2)
        self.head_out = KANLayer(
            d_in=d_model // 2, d_out=1,
            grid_size=grid_size, spline_order=spline_order,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_time_window(self, t):
        t_seq = t.unsqueeze(1) + self.time_offsets
        return torch.clamp(t_seq, min=0.0)

    def _encode_spatial(self, r_s):
        # 空间傅里叶特征 + 空间KAN编码
        proj_s = r_s @ self.B_spatial + self.b_spatial
        phi_s = torch.cat([
            torch.sin(2 * self.pi * proj_s),
            torch.cos(2 * self.pi * proj_s),
            r_s,
        ], dim=-1)

        s_feat = phi_s
        for kan, norm in zip(self.spatial_kan_layers, self.spatial_norms):
            s_feat = norm(kan(s_feat))

        return s_feat

    def _encode_temporal_tokens(self, t):
        # 时间傅里叶特征 + BiLSTM时序
        B = t.shape[0]

        t_seq = self._build_time_window(t)
        t_flat = t_seq.reshape(B * self.seq_len, 1)

        proj_t = t_flat @ self.B_temporal + self.b_temporal
        phi_t_flat = torch.cat([
            torch.sin(2 * self.pi * proj_t),
            torch.cos(2 * self.pi * proj_t),
            t_flat,
        ], dim=-1)
        phi_t = phi_t_flat.reshape(B, self.seq_len, -1)
        phi_t_2d = phi_t.reshape(B * self.seq_len, -1)
        X_t_flat = self.temporal_embed(phi_t_2d)
        X_t = X_t_flat.reshape(B, self.seq_len, self.d_model)
        X_t = X_t + self.temporal_pos_emb

        if self.use_lstm:
            H, _ = self.temporal_lstm(X_t)
            t_tokens = self.lstm_proj(H)
        else:
            t_tokens = self.temporal_token_proj(X_t)

        return t_tokens

    def _cross_attention_fusion(self, s_feat, t_tokens, return_attention=False):
        q = s_feat.unsqueeze(1)  # [B, 1, d_model]
        attn_maps = []

        for attn_layer, attn_norm, ffn, ffn_norm in zip(
            self.attn_layers,
            self.attn_norms,
            self.ffn_layers,
            self.ffn_norms,
        ):
            attn_out, attn_w = attn_layer(
                q, t_tokens, t_tokens,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            if return_attention:
                attn_maps.append(attn_w.squeeze(2))  # [B, num_heads, seq_len]
            q = attn_norm(q + attn_out)

            ffn_out = ffn(q)
            q = ffn_norm(q + ffn_out)

        fused = q.squeeze(1)  # [B, d_model]

        if return_attention:
            return fused, torch.stack(attn_maps, dim=1)  # [B, n_layers, num_heads, seq_len]
        return fused

    def _pooling_fusion(self, s_feat, t_tokens):
        if self.temporal_pool == "last":
            t_feat = t_tokens[:, -1, :]
        elif self.temporal_pool == "mean":
            t_feat = t_tokens.mean(dim=1)
        else:
            raise ValueError(f"Unknown temporal_pool={self.temporal_pool!r}")

        return self.fusion_proj(torch.cat([s_feat, t_feat], dim=-1))

    def forward(self, r: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Args:  r: [B, 3] (x, y, t)
        Returns: out: [B, 1]
        """
        r_s = r[:, :2]
        t = r[:, 2:3]
        
        # [A] 空间傅里叶特征 + KAN编码
        s_feat = self._encode_spatial(r_s)
        
        # [B] 时间傅里叶特征 + BiLSTM时序/轻量级MLP
        t_tokens = self._encode_temporal_tokens(t)

        # [C] 交叉注意力/池化融合
        if self.use_attention:
            if return_attention:
                fused, attn_maps = self._cross_attention_fusion(
                    s_feat, t_tokens, return_attention=True
                )
            else:
                fused = self._cross_attention_fusion(s_feat, t_tokens)
                attn_maps = None
        else:
            fused = self._pooling_fusion(s_feat, t_tokens)
            attn_maps = None
        
        # [D] KAN输出头
        h = self.head_norm(self.head_kan(fused))
        out = self.head_out(h)
        if return_attention:
            return out, {
                "attention": attn_maps,
                "time_offsets": self.time_offsets.squeeze(0).squeeze(-1),
            }
        return out

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        spline_params = 0
        base_params = 0
        fourier_params = 0
        lstm_params = 0
        attn_params = 0
        fusion_params = 0
        other_params = 0

        for name, p in self.named_parameters():
            n = p.numel()
            if 'spline' in name or 'base_weight' in name or 'spline_scaler' in name:
                spline_params += n
            elif 'B_spatial' in name or 'B_temporal' in name or 'b_spatial' in name or 'b_temporal' in name:
                fourier_params += n
            elif 'lstm' in name:
                lstm_params += n
            elif 'attn' in name or 'ffn' in name:
                attn_params += n
            elif "fusion_proj" in name or "temporal_token_proj" in name:
                fusion_params += n
            else:
                other_params += n
        return {
            'total': total,
            "temporal_mode": self.temporal_mode,
            'kan(spline+base)': spline_params,
            'fourier': fourier_params,
            'lstm': lstm_params,
            'attention+ffn': attn_params,
            "fusion/token_proj": fusion_params,
            'other': other_params,
        }
