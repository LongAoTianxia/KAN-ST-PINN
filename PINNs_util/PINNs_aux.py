import torch 
import torch.nn as nn
import numpy as np


class SpaceTimeAttentionLSTM(nn.Module):
    """
    Space-Time Attention LSTM for PINNs
    
    1. 引入时间窗口机制，从当前时刻 t 构造过去时间窗口 [t-(n_t-1)*dt, ..., t-dt, t]
    每个时间步独立编码，共享投影层，添加时间位置编码，显式建模时间顺序

    2. Fourier feature Embedding 

    3. L个 Factorized ST Blocks:
       - 在 block 前构造点内 token 网格 Z: [B, n_t, n_s, d_tok]
       - (A) Temporal Modeling: 对每个空间 index s，沿 n_t 做 BiLSTM
       - (B) Temporal Attention: 沿 n_t 轴加权
       - (C) Spatial MHSA: 对每个时间 index t，沿 n_s 做 MHSA
       - (D) FFN + Residual + LayerNorm
       - 在最后一层 block 输出拆分回 X_s, X_t

    4. Pointwise Head: pool/flatten 后输出 p
    
    Args:
        n_spatial: 空间坐标维度 (2 for x,y)
        n_temporal: 时间坐标维度 (1 for t)
        n_out: 输出维度
        d_tok: 每个 token 的嵌入维度
        n_s: 空间 token 数量
        n_t: 时间 token 数量
        n_ffeatures: 傅里叶特征数量 (0表示不使用)
        lstm_hidden: BiLSTM 隐藏维度
        lstm_layers: BiLSTM 层数
        num_heads: 注意力头数
        n_blocks: ST Block 数量
        mlp_ratio: FFN 中间层比例
        dropout: Dropout 比例
        dt_token: 时间窗口步长（默认 None 时自动设为 0.01）
        
    d_tok % num_heads == 0
    """
    
    def __init__(self, n_spatial=2, n_temporal=1, n_out=1, d_tok=16, n_s=2, n_t=2,
                 n_ffeatures=64, lstm_hidden=16, lstm_layers=1, num_heads=4,
                 n_blocks=2, mlp_ratio=4., dropout=0.0, dt_token=None,
                 use_longhistory_temporal=True, disable_block_temporal=True):
        super().__init__()
        
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.n_out = n_out
        self.d_tok = d_tok
        self.n_s = n_s
        self.n_t = n_t
        self.n_ffeatures = n_ffeatures
        self.hidden_dim = (n_s + n_t) * d_tok  # 用于兼容外部接口
        self.use_longhistory_temporal = use_longhistory_temporal
        self.disable_block_temporal = disable_block_temporal

        # 时间窗口步长
        self.dt_token = dt_token if dt_token is not None else 0.01

        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
        # ============ 1. Fourier Feature Embedding ============
        if n_ffeatures != 0:
            # 空间傅里叶特征
            self.B_spatial = nn.Parameter(torch.randn((n_spatial, n_ffeatures)))
            self.b_spatial = nn.Parameter(torch.randn((1, n_ffeatures)))
            # 时间傅里叶特征（单个时间标量）
            self.B_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))  # 改为 (1, n_ffeatures)
            self.b_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
            d_s = n_ffeatures  # 空间特征维度
            d_t = n_ffeatures  # 单个时间步的特征维度
        else:
            d_s = n_spatial
            d_t = 1  # 单个时间标量
        
        # ============ 2. 投影层: 映射到 token 序列 ============
        # 空间投影: [B, d_s] -> [B, n_s * d_tok] -> reshape -> [B, n_s, d_tok]
        self.spatial_proj = nn.Linear(d_s, n_s * d_tok)
        # 时间投影: [B, d_t] -> [B, d_tok]（每个时间步共享，逐 token 投影）
        self.temporal_proj = nn.Linear(d_t, d_tok)
        
        # 时间位置编码
        # 为 n_t 个时间步提供可学习的位置信息
        time_offsets = -torch.arange(n_t - 1, -1, -1, dtype=torch.float32) * self.dt_token
        self.register_buffer("time_offsets", time_offsets.view(1, n_t, 1))

        # 为 n_t 个时间步提供可学习的位置信息
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, n_t, d_tok))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # >>> LEGACY_TEMPORAL_PATH_DISABLED_BEGIN (kept for possible restore)
        # 旧路径中主要时间建模发生在 FactorizedSTBlock 内部。
        # <<< LEGACY_TEMPORAL_PATH_DISABLED_END
        # 新路径：在 block 前采用 LongHistory 风格时间编码，并保留 temporal self-attention。
        if self.use_longhistory_temporal:
            self.temporal_lstm = nn.LSTM(
                input_size=d_tok,
                hidden_size=d_tok,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            self.temporal_lstm_norm = nn.LayerNorm(d_tok)
            self.temporal_attn_norm1 = nn.LayerNorm(d_tok)
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=d_tok,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_attn_norm2 = nn.LayerNorm(d_tok)

        # ============ 3. Factorized ST Blocks ============
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(
                d_tok=d_tok,
                n_s=n_s,
                n_t=n_t,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                is_first=(i == 0),  # 标记第一个 block
                disable_temporal_mixer=disable_block_temporal
            ) for i in range(n_blocks)
        ])
        
        # ============ 4. Pointwise Head ============
        # 融合后的维度: (n_s + n_t) * d_tok
        # 旧版拼接融合(已弃用)
        #fusion_dim = (n_s + n_t) * d_tok
        #self.head = nn.Sequential(
        #    nn.LayerNorm(fusion_dim),
        #    nn.Linear(fusion_dim, fusion_dim // 2),
        #    nn.Tanh(),
        #    nn.Linear(fusion_dim // 2, n_out)
        #)

        # 联合 attention pooling: 对 n_t * n_s 个 token 做加权池化
        # w^T * phi(Z_i) -> softmax -> 加权求和 -> z_pool: [B, d_tok]
        self.pool_phi = nn.Sequential(
            nn.Linear(d_tok, d_tok),
            nn.Tanh()
        )
        self.pool_w = nn.Linear(d_tok, 1, bias=False)  # w^T
        
        # Head 输入维度改为 d_tok（池化后单个向量）
        self.head = nn.Sequential(
            nn.LayerNorm(d_tok),
            nn.Linear(d_tok, d_tok // 2),
            nn.Tanh(),
            nn.Linear(d_tok // 2, n_out)
        )
        # 保持 hidden_dim 兼容外部接口（可选保留）
        self.hidden_dim = (n_s + n_t) * d_tok

    def _build_past_time_window(self, t_cur):
        """
        构造过去时间窗口（包含当前时刻）
        
        Args:
            t_cur: [B, 1] 当前时间点
        Returns:
            t_seq: [B, n_t, 1] 时间窗口序列
                   [t - (n_t-1)*dt, t - (n_t-2)*dt, ..., t - dt, t]
        """
        # 使用 buffer 中的 time_offsets: [1, n_t, 1]
        # t_cur: [B, 1, 1] + time_offsets: [1, n_t, 1] -> [B, n_t, 1]
        t_seq = t_cur.unsqueeze(1) + self.time_offsets  # [B, n_t, 1]
        
        # 截断负值到 0（确保物理时间非负）
        t_seq = torch.clamp(t_seq, min=0.0)
        
        return t_seq
    
    def forward(self, r):
        """
        Args:
            r: [B, 3] 输入坐标 (x, y, t)
        Returns:
            out: [B, n_out] 预测值
        """
        B = r.shape[0]
        
        # ---- Step 0: 分离空间和时间坐标 ----
        r_s = r[:, :self.n_spatial]   # [B, n_spatial] (x, y)
        r_t = r[:, self.n_spatial:self.n_spatial+1]   # [B, 1] (t)
        
        # ---- Step 1: 构造时间窗口序列 ----
        t_seq = self._build_past_time_window(r_t)  # [B, n_t, 1]

        # ---- Step 2: Fourier Feature Embedding ----
        if self.n_ffeatures != 0:
            # 空间特征（所有时间步共享）
            phi_s = torch.cos(2 * self.pi * r_s @ self.B_spatial + self.b_spatial)  # [B, n_ffeatures]
            
            # 时间特征（每个时间步独立编码）
            # t_seq: [B, n_t, 1] -> reshape -> [B*n_t, 1]
            t_seq_flat = t_seq.reshape(B * self.n_t, 1)  # [B*n_t, 1]
            phi_t_flat = torch.cos(2 * self.pi * t_seq_flat @ self.B_temporal + self.b_temporal)  # [B*n_t, n_ffeatures]
            phi_t = phi_t_flat.reshape(B, self.n_t, self.n_ffeatures)  # [B, n_t, n_ffeatures]
        else:
            phi_s = r_s  # [B, n_spatial]
            phi_t = t_seq  # [B, n_t, 1]
        
        # ---- Step 3: 投影到 token 序列 ----
        # 空间 token（所有时间步共享）
        u_s = self.spatial_proj(phi_s)  # [B, n_s * d_tok]
        X_s = u_s.reshape(B, self.n_s, self.d_tok)  # [B, n_s, d_tok]
        
        # 时间 token（逐 token 投影）
        # phi_t展平: [B, n_t, d_t] -> [B*n_t, d_t]
        phi_t_flat = phi_t.reshape(B * self.n_t, -1)  # [B*n_t, d_t]
        X_t_flat = self.temporal_proj(phi_t_flat)  # [B*n_t, d_tok]
        X_t = X_t_flat.reshape(B, self.n_t, self.d_tok)  # [B, n_t, d_tok]
        
        # 添加时间位置编码
        X_t = X_t + self.temporal_pos_embed  # [B, n_t, d_tok]

        # LongHistory 风格：先做时间序列编码，再进入时空 block
        if self.use_longhistory_temporal:
            H_t, _ = self.temporal_lstm(X_t)
            H_t = self.temporal_lstm_norm(X_t + H_t)
            H_t_norm = self.temporal_attn_norm1(H_t)
            H_t_attn, _ = self.temporal_attn(H_t_norm, H_t_norm, H_t_norm)
            X_t = self.temporal_attn_norm2(H_t + H_t_attn)
        
        # ---- Step 4: L个 Factorized ST Blocks ----
        # 第1个 block: 输入 (X_s, X_t), 输出 Z
        # 后续 block: 输入输出都是 Z
        Z = None
        for i, block in enumerate(self.blocks):
            if i == 0:
                Z = block(X_s, X_t, Z_in=None)  # 第一个 block 构造 Z
            else:
                Z = block(None, None, Z_in=Z)   # 后续 block 直接处理 Z
        

        # Step 5.1: flatten 时空维度
        Z_flat = Z.reshape(B, self.n_t * self.n_s, self.d_tok)  # [B, n_t*n_s, d_tok]
        
        # Step 5.2: 计算注意力权重 alpha_i = softmax(w^T * phi(Z_i))
        phi_Z = self.pool_phi(Z_flat)           # [B, n_t*n_s, d_tok]
        scores = self.pool_w(phi_Z)             # [B, n_t*n_s, 1]
        alpha = torch.softmax(scores, dim=1)    # [B, n_t*n_s, 1]
        
        # Step 5.3: 加权求和 z_pool = sum_i(alpha_i * Z_i)
        z_pool = (alpha * Z_flat).sum(dim=1)    # [B, d_tok]
        
        # Step 5.4: 输出
        out = self.head(z_pool)                 # [B, n_out]
        

        return out


class FactorizedSTBlock(nn.Module):

    
    def __init__(self, d_tok, n_s, n_t, lstm_hidden, lstm_layers=1,
                 num_heads=4, mlp_ratio=4., dropout=0.0, is_first=False,
                 disable_temporal_mixer=False):
        super().__init__()
        
        self.d_tok = d_tok
        self.n_s = n_s
        self.n_t = n_t
        self.is_first = is_first
        self.disable_temporal_mixer = disable_temporal_mixer

        if is_first:
            self.st_fusion_proj = nn.Linear(2 * d_tok, d_tok)

        if not self.disable_temporal_mixer:
            self.temporal_lstm = nn.LSTM(
                input_size=d_tok,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0.0,
                bidirectional=False
            )
            # BiLSTM 输出维度是 lstm_hidden，投影回 d_tok
            self.temporal_proj_back = nn.Linear(lstm_hidden, d_tok)
            self.temporal_norm = nn.LayerNorm(d_tok)
            
            # 逐 token 加权不聚合
            self.temporal_attn_gate = nn.Sequential(
                nn.Linear(d_tok, d_tok),
                nn.Tanh(),
                nn.Linear(d_tok, d_tok),
                nn.Sigmoid()  # 每个 token 每个维度独立门控 [0, 1]
            )
            self.temporal_attn_norm = nn.LayerNorm(d_tok)
        
        self.spatial_norm1 = nn.LayerNorm(d_tok)
        self.spatial_mhsa = nn.MultiheadAttention(
            embed_dim=d_tok,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.spatial_norm2 = nn.LayerNorm(d_tok)
        self.ffn = nn.Sequential(
            nn.Linear(d_tok, int(d_tok * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_tok * mlp_ratio), d_tok),
            nn.Dropout(dropout)
        ) 

        self.block_norm = nn.LayerNorm(d_tok)
    def forward(self, X_s, X_t, Z_in):
        """
        Args:
            X_s: [B, n_s, d_tok] 空间 token 序列
            X_t: [B, n_t, d_tok] 时间 token 序列
            Z_in: [B, n_t, n_s, d_tok] 输入的时空网格
        Returns:
            Z_out [B, n_t, n_s, d_tok]
        """

        if self.is_first:
            assert X_s is not None, "第1个block：X_s 不能为 None"
            assert X_t is not None, "第1个block：X_t 不能为 None"
        else:
            assert Z_in is not None, f"非第1个block：Z_in 不能为 None"
            assert Z_in.shape == (Z_in.shape[0], self.n_t, self.n_s, self.d_tok), \
                f"Z_in 形状错误：期望 (B, {self.n_t}, {self.n_s}, {self.d_tok})，实际 {Z_in.shape}"



        if self.is_first:
            B = X_s.shape[0]
            # X_t: [B, n_t, d_tok] -> [B, n_t, 1, d_tok] -> broadcast -> [B, n_t, n_s, d_tok]
            # X_s: [B, n_s, d_tok] -> [B, 1, n_s, d_tok] -> broadcast -> [B, n_t, n_s, d_tok]
            X_t_expanded = X_t.unsqueeze(2).expand(B, self.n_t, self.n_s, self.d_tok)
            X_s_expanded = X_s.unsqueeze(1).expand(B, self.n_t, self.n_s, self.d_tok)
            Z_concat = torch.cat([X_t_expanded, X_s_expanded], dim=-1)  # [B, n_t, n_s, 2*d_tok]
            Z = self.st_fusion_proj(Z_concat)  # [B, n_t, n_s, d_tok]
            Z_residual = Z 
        else:
            Z = Z_in
            B = Z.shape[0]  # 从 Z_in 推断 batch size
            Z_residual = Z_in

        if self.disable_temporal_mixer:
            # 时间分支前置到 SpaceTimeAttentionLSTM；block 内仅保留空间 attention + FFN。
            H_t_attn = Z
        else:
            # reshape: [B, n_t, n_s, d_tok] -> [B*n_s, n_t, d_tok]
            Z_reshaped = Z.permute(0, 2, 1, 3).contiguous().reshape(B * self.n_s, self.n_t, self.d_tok)
            
            # BiLSTM
            H_t, _ = self.temporal_lstm(Z_reshaped)  # [B*n_s, n_t, lstm_hidden]
            H_t_proj = self.temporal_proj_back(H_t)  # [B*n_s, n_t, d_tok]
            
            # 残差 + LayerNorm
            H_t_res = self.temporal_norm(Z_reshaped + H_t_proj)  # [B*n_s, n_t, d_tok]
            
            gate = self.temporal_attn_gate(H_t_res)  # [B*n_s, n_t, d_tok]
            H_t_attn = gate * H_t_res                # 逐元素门控
            H_t_attn = self.temporal_attn_norm(H_t_res + H_t_attn)  # [B*n_s, n_t, d_tok]
            
            # reshape 回: [B, n_s, n_t, d_tok] -> [B, n_t, n_s, d_tok]
            H_t_attn = H_t_attn.reshape(B, self.n_s, self.n_t, self.d_tok).permute(0, 2, 1, 3)
        
        # reshape: [B, n_t, n_s, d_tok] -> [B*n_t, n_s, d_tok]
        H_s_input = H_t_attn.reshape(B * self.n_t, self.n_s, self.d_tok)
        
        # Pre-Norm + MHSA
        H_s_normed = self.spatial_norm1(H_s_input)  # [B*n_t, n_s, d_tok]
        H_s_attn, _ = self.spatial_mhsa(H_s_normed, H_s_normed, H_s_normed)  # [B*n_t, n_s, d_tok]
        
        # 残差
        H_s = H_s_input + H_s_attn  # [B*n_t, n_s, d_tok]
        
        H_s_normed = self.spatial_norm2(H_s)
        H_s_out = H_s + self.ffn(H_s_normed)  # [B*n_t, n_s, d_tok]
        
        Z_processed = H_s_out.reshape(B, self.n_t, self.n_s, self.d_tok)
        

        Z_out = self.block_norm(Z_residual + Z_processed)  # [B, n_t, n_s, d_tok]

        return Z_out
    
class LSTMAttention(nn.Module):
    """LSTM+注意力模块"""
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers=1, dropout=0.0,
                 use_attention=True, seq_mode='dim', n_ffeatures=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.seq_mode = seq_mode
        self.input_dim = input_dim
        self.n_out = output_dim
        self.raw_input_dim = input_dim
        self.n_ffeatures = n_ffeatures
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        if seq_mode == 'dim':
            if n_ffeatures != 0:
                self.B = nn.Parameter(torch.randn((input_dim, n_ffeatures)))
                self.b = nn.Parameter(torch.randn((1, n_ffeatures)))
                self.input_dim = n_ffeatures
            proj_dim = max(1, hidden_dim // 2)
            self.input_proj = nn.Linear(1, proj_dim)
            lstm_input_dim = proj_dim
        elif seq_mode == 'spatial_temporal':
            proj_dim = max(1, hidden_dim // 2)
            if n_ffeatures != 0:
                if input_dim > 1:
                    self.B_spatial = nn.Parameter(torch.randn((input_dim - 1, n_ffeatures)))
                    self.b_spatial = nn.Parameter(torch.randn((1, n_ffeatures)))
                else:
                    self.B_spatial = None
                self.B_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
                self.b_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
                spatial_in = n_ffeatures if input_dim > 1 else 0
                temporal_in = n_ffeatures
            else:
                spatial_in = input_dim - 1 if input_dim > 1 else 0
                temporal_in = 1
            if input_dim > 1:
                self.spatial_proj = nn.Linear(spatial_in, proj_dim)
            else:
                self.spatial_proj = None
            self.temporal_proj = nn.Linear(temporal_in, proj_dim)
            lstm_input_dim = proj_dim
        else:
            if n_ffeatures != 0:
                self.B = nn.Parameter(torch.randn((input_dim, n_ffeatures)))
                self.b = nn.Parameter(torch.randn((1, n_ffeatures)))
                self.input_dim = n_ffeatures
            lstm_input_dim = self.input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, self.n_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim]
        Returns:
            out: [B, hidden_dim]
        """
        if self.seq_mode == 'dim':
            if self.n_ffeatures != 0:
                x = torch.cos(2 * self.pi * x @ self.B + self.b)
            x_seq = x.unsqueeze(-1)
            x_seq = self.input_proj(x_seq)
        elif self.seq_mode == 'spatial_temporal':
            if self.raw_input_dim > 1:
                spatial = x[:, :-1]
                temporal = x[:, -1:]
                if self.n_ffeatures != 0:
                    if self.B_spatial is not None:
                        spatial = torch.cos(2 * self.pi * spatial @ self.B_spatial + self.b_spatial)
                    temporal = torch.cos(2 * self.pi * temporal @ self.B_temporal + self.b_temporal)
                spatial_feat = self.spatial_proj(spatial).unsqueeze(1)
                temporal_feat = self.temporal_proj(temporal).unsqueeze(1)
                x_seq = torch.cat([spatial_feat, temporal_feat], dim=1)
            else:
                temporal = x
                if self.n_ffeatures != 0:
                    temporal = torch.cos(2 * self.pi * temporal @ self.B_temporal + self.b_temporal)
                temporal_feat = self.temporal_proj(temporal).unsqueeze(1)
                x_seq = temporal_feat
        else:
            if self.n_ffeatures != 0:
                x = torch.cos(2 * self.pi * x @ self.B + self.b)
            x_seq = x.unsqueeze(1)

        lstm_out, (h_n, c_n) = self.lstm(x_seq)

        if self.use_attention:
            query = lstm_out[:, -1:, :]
            attn_out, attn_weights = self.attention(query, lstm_out, lstm_out)
            out = self.norm(attn_out.squeeze(1))
        else:
            out = lstm_out[:, -1, :]

        out = self.output_proj(out)
        out = self.dropout(out)

        return out


def pde_residual(p, r, c):
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_xx = torch.autograd.grad(p_r[:,0], r, torch.ones_like(p_r[:,0]), create_graph=True)[0][:,0:1]
    p_yy = torch.autograd.grad(p_r[:,1], r, torch.ones_like(p_r[:,1]), create_graph=True)[0][:,1:2]
    p_tt = torch.autograd.grad(p_r[:,2], r, torch.ones_like(p_r[:,2]), create_graph=True)[0][:,2:3]
    pde_res = p_xx + p_yy - p_tt/c**2
    return pde_res

def loss_grad_norm(loss, model):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(torch.sum(g**2) for g in grads if g is not None)
    return torch.sqrt(total).detach()
def update_lambda(model, loss_lst, lamb_lst, alpha):
    grad = []
    for loss in loss_lst:
        grad.append(loss_grad_norm(loss, model))
    grad_sum = sum(grad)
    lamb = []
    for i in range(len(grad)):
        lamb_hat = grad_sum / grad[i]
        if torch.isnan(lamb_hat) or torch.isinf(lamb_hat):
            lamb_hat = torch.ones_like(lamb_hat)
        # 添加上下界约束，防止爆炸
        if i == 2:  
            lamb_hat = torch.clamp(lamb_hat, min=0.0001, max=1e3)  # bc
        else:
            lamb_hat = torch.clamp(lamb_hat, min=0.0001, max=1e4)  # 控制范围
        
        lamb_new = alpha*lamb_lst[i] + (1-alpha)*lamb_hat
        lamb.append(lamb_new)
    return lamb

def xyt_tensor(rxy, t, device):
    n_t = len(t)
    n_xy = len(rxy)
    r = np.column_stack(
        (np.repeat(rxy, n_t, axis=0),
         np.tile(t, n_xy)))
    r = torch.tensor(r).view(-1,3).requires_grad_(True)
    r = r.to(device)
    return r

def rand_colloc(n_colloc, L, T, device):
    dims_domain = torch.tensor((L,L,T), device=device)
    dims_domain = torch.reshape(dims_domain, (1,3))
    r_colloc = dims_domain*torch.randn((n_colloc,3), device=device).requires_grad_(True)
    #xy = torch.rand((n_colloc, 2), device=device) * L      # [0, L]
    #tt = torch.rand((n_colloc, 1), device=device) * T      # [0, T]
    #r_colloc = torch.cat([xy, tt], dim=1).requires_grad_(True)
    return r_colloc

def rand_colloc_fixed(n_colloc, L, T, device, ratio_gaussian=0.85):
    n_gauss = int(n_colloc * ratio_gaussian)
    n_uniform = n_colloc - n_gauss
    
    # 高斯部分 - 改进版
    # 空间: 中心在 (L/2, L/2)，标准差为 L/6 使大部分点落在 [0,L] 内
    xy_center = L / 2
    xy_std = L / 6  # 3σ ≈ L/2, 保证大部分点在 [0, L] 范围内
    xy_gauss = xy_center + xy_std * torch.randn((n_gauss, 2), device=device)
    xy_gauss = torch.clamp(xy_gauss, 0.0, L)
    # 时间: 取绝对值确保 t >= 0，同时自然增强 t 接近 0 处的采样密度
    t_std = T / 3  # 标准差，可调节
    t_gauss = torch.abs(t_std * torch.randn((n_gauss, 1), device=device))
    t_gauss  = torch.clamp(t_gauss,  0.0, T)
    r_gauss = torch.cat([xy_gauss, t_gauss], dim=1)
    
    # 均匀部分（保持不变，确保边界区域覆盖）
    xy = torch.rand((n_uniform, 2), device=device) * L
    tt = torch.rand((n_uniform, 1), device=device) * T
    r_uniform = torch.cat([xy, tt], dim=1)
    
    r_colloc = torch.cat([r_gauss, r_uniform], dim=0).requires_grad_(True)
    return r_colloc

def rand_colloc_mixed(n_colloc, L, T, device, ratio_gaussian=0.85):
    n_gauss = int(n_colloc * ratio_gaussian)
    n_uniform = n_colloc - n_gauss
    
    # 高斯部分
    dims = torch.tensor((L, L, T), device=device).reshape(1, 3)
    r_gauss = dims * torch.randn((n_gauss, 3), device=device)
    
    # 均匀部分
    xy = torch.rand((n_uniform, 2), device=device) * L
    tt = torch.rand((n_uniform, 1), device=device) * T
    r_uniform = torch.cat([xy, tt], dim=1)
    
    r_colloc = torch.cat([r_gauss, r_uniform], dim=0).requires_grad_(True)
    return r_colloc

def rand_boundary(n_bc, L, t_crr, device):
    # 随机决定每个点属于的边  0:x=0, 1:x=L, 2:y=0, 3:y=L
    side = torch.randint(0, 4, (n_bc,1), device=device)
    # 生成[0,1]间的随机xy坐标值
    xy = torch.rand((n_bc,2), device=device)
    xy = xy * L
    # 在每一个边界点生成一个t_crr间的随机时间值
    tt = torch.rand((n_bc,1), device=device) * t_crr

    x = xy[:,0:1]; y = xy[:,1:2]
    x = torch.where(side==0, torch.zeros_like(x), x)  
    x = torch.where(side==1, L*torch.ones_like(x), x)
    y = torch.where(side==2, torch.zeros_like(y), y)
    y = torch.where(side==3, L*torch.ones_like(y), y)
    # 拼接成边界点坐标(x, y, t),并启用 autograd
    r = torch.cat([x,y,tt], dim=1).requires_grad_(True)

    return r, side


def bc_residual_absorbing(p, r, side, c):
    """
    计算吸收边界条件的残差
    吸收边界条件 (基于 Sommerfeld/Mur 一阶吸收边界):
        x=0 边界: u_x - u_t / c =0  (向-x传播的波被吸收)
        x=L 边界: u_x +  u_t / c = 0 (向x传播的波被吸收)
        y=0 边界: u_y - u_t / c = 0  (向-y传播的波被吸收)
        y=L 边界: u_y +  u_t / c = 0  (向y传播的波被吸收)
    
    Args:
        p: 模型预测的波场值 [n_bc, 1]
        r: 边界点坐标 [n_bc, 3] (x, y, t)，需要 requires_grad=True
        side: 边界标识 [n_bc, 1]，0:x=0, 1:x=L, 2:y=0, 3:y=L
        c: 边界点处的波速 [n_bc, 1]
    
    Returns:
        bc_res: 边界条件残差 [n_bc, 1]
    """
    # 计算 p 对 r 的梯度: [p_x, p_y, p_t]
    p_r = torch.autograd.grad(p, r, torch.ones_like(p), create_graph=True)[0]
    p_x = p_r[:, 0:1]  # ∂p/∂x
    p_y = p_r[:, 1:2]  # ∂p/∂y
    p_t = p_r[:, 2:3]  # ∂p/∂t
    
    bc_res = torch.zeros_like(p)
    
    mask_x0 = (side == 0).squeeze()
    mask_xL = (side == 1).squeeze()
    mask_y0 = (side == 2).squeeze()
    mask_yL = (side == 3).squeeze()
    
    if mask_x0.any():
        bc_res[mask_x0] = p_x[mask_x0] * c[mask_x0] - p_t[mask_x0]
    if mask_xL.any():
        bc_res[mask_xL] = p_x[mask_xL] * c[mask_xL] + p_t[mask_xL] 
    if mask_y0.any():
        bc_res[mask_y0] = p_y[mask_y0] * c[mask_y0] - p_t[mask_y0] 
    if mask_yL.any():
        bc_res[mask_yL] = p_y[mask_yL] * c[mask_yL] + p_t[mask_yL] 
    
    return bc_res



class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子 1/sqrt(d_k)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: [B, N, C] 其中N是token数量(空间+时间)，C是嵌入维度
        B, N, C = x.shape
        
        # qkv: [B, N, 3*C] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # [B, num_heads, N, head_dim] -> [B, N, C]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    交叉注意力模块：用于空间和时间特征的融合
    query来自一个分支，key/value来自另一个分支
    """
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value):
        # query: [B, 1, C], key_value: [B, 1, C]
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FCN_Attention(nn.Module):
    """
    带注意力机制的FCN网络
    将输入r分为空间(x,y,...)和时间t两条支路，各自MLP编码，再用注意力融合
    
    Args:
        n_spatial: 空间维度数量 (例如2表示x,y)
        n_temporal: 时间维度数量 (通常为1)
        n_out: 输出维度
        embed_dim: 嵌入维度 (需要能被num_heads整除)
        n_hidden: MLP隐藏层维度
        n_layers: 每个分支的MLP层数
        num_heads: 注意力头数量
        n_ffeatures: 傅里叶特征数量 (0表示不使用)
        mlp_ratio: MLP中间层维度比例
        drop_ratio: Dropout比例
        fusion_mode: 融合模式 ('self_attn', 'cross_attn', 'concat_attn')
    """
    def __init__(self, n_spatial=2, n_temporal=1, n_out=1, embed_dim=64, n_hidden=128, 
                 n_layers=3, num_heads=4, n_ffeatures=0, mlp_ratio=4., drop_ratio=0.,
                 fusion_mode='self_attn'):
        super().__init__()
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.n_ffeatures = n_ffeatures
        self.n_spatial = n_spatial
        self.n_temporal = n_temporal
        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode
        
        # 傅里叶特征映射 (可选)
        if n_ffeatures != 0:
            self.B_spatial = nn.Parameter(torch.randn((n_spatial, n_ffeatures)))
            self.b_spatial = nn.Parameter(torch.randn((1, n_ffeatures)))
            self.B_temporal = nn.Parameter(torch.randn((n_temporal, n_ffeatures)))
            self.b_temporal = nn.Parameter(torch.randn((1, n_ffeatures)))
            spatial_in = n_ffeatures
            temporal_in = n_ffeatures
        else:
            spatial_in = n_spatial
            temporal_in = n_temporal
        
        # 空间分支MLP编码器
        activation = nn.Tanh
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_in, n_hidden),
            activation(),
            *[nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers - 1)],
            nn.Linear(n_hidden, embed_dim)
        )
        
        # 时间分支MLP编码器
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_in, n_hidden),
            activation(),
            *[nn.Sequential(nn.Linear(n_hidden, n_hidden), activation()) for _ in range(n_layers - 1)],
            nn.Linear(n_hidden, embed_dim)
        )
        
        # 位置编码 (用于区分空间和时间token)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        # 注意力融合模块
        if fusion_mode == 'self_attn':
            # 自注意力融合: 将空间和时间token拼接后做自注意力
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attention = Attention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, drop=drop_ratio)
            fusion_out_dim = embed_dim * 2  # 拼接两个token
            
        elif fusion_mode == 'cross_attn':
            # 交叉注意力融合: 空间query时间，时间query空间
            self.norm_s = nn.LayerNorm(embed_dim)
            self.norm_t = nn.LayerNorm(embed_dim)
            self.cross_attn_s2t = CrossAttention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.cross_attn_t2s = CrossAttention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.norm_out = nn.LayerNorm(embed_dim * 2)
            self.mlp = MLP(embed_dim * 2, int(embed_dim * 2 * mlp_ratio), embed_dim * 2, drop=drop_ratio)
            fusion_out_dim = embed_dim * 2
            
        elif fusion_mode == 'concat_attn':
            # 简单拼接后通过注意力层
            self.norm1 = nn.LayerNorm(embed_dim * 2)
            self.proj = nn.Linear(embed_dim * 2, embed_dim)
            self.attention = Attention(embed_dim, num_heads=num_heads, attn_drop=drop_ratio, proj_drop=drop_ratio)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, drop=drop_ratio)
            fusion_out_dim = embed_dim
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(fusion_out_dim),
            nn.Linear(fusion_out_dim, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, r):
        # r: [B, n_spatial + n_temporal] 例如 [B, 3] 表示 (x, y, t)
        B = r.shape[0]
        
        # 分离空间和时间坐标
        r_spatial = r[:, :self.n_spatial]   # [B, n_spatial]
        r_temporal = r[:, self.n_spatial:]  # [B, n_temporal]
        
        # 傅里叶特征映射 (可选)
        if self.n_ffeatures != 0:
            r_spatial = torch.cos(2 * self.pi * r_spatial @ self.B_spatial + self.b_spatial)
            r_temporal = torch.cos(2 * self.pi * r_temporal @ self.B_temporal + self.b_temporal)
        
        # 分支编码
        spatial_feat = self.spatial_encoder(r_spatial)    # [B, embed_dim]
        temporal_feat = self.temporal_encoder(r_temporal)  # [B, embed_dim]
        
        # 添加位置编码，转换为token形式 [B, 1, embed_dim]
        spatial_token = spatial_feat.unsqueeze(1) + self.spatial_pos_embed
        temporal_token = temporal_feat.unsqueeze(1) + self.temporal_pos_embed
        
        # 注意力融合
        if self.fusion_mode == 'self_attn':
            # 拼接tokens: [B, 2, embed_dim]
            tokens = torch.cat([spatial_token, temporal_token], dim=1)
            # 自注意力 + 残差
            tokens = tokens + self.attention(self.norm1(tokens))
            tokens = tokens + self.mlp(self.norm2(tokens))
            # 展平: [B, 2*embed_dim]
            out = tokens.flatten(1)
            
        elif self.fusion_mode == 'cross_attn':
            # 交叉注意力: 空间query时间, 时间query空间
            spatial_out = spatial_token + self.cross_attn_s2t(self.norm_s(spatial_token), self.norm_t(temporal_token))
            temporal_out = temporal_token + self.cross_attn_t2s(self.norm_t(temporal_token), self.norm_s(spatial_token))
            # 拼接并通过MLP
            out = torch.cat([spatial_out, temporal_out], dim=-1)  # [B, 1, 2*embed_dim]
            out = out + self.mlp(self.norm_out(out))
            out = out.squeeze(1)  # [B, 2*embed_dim]
            
        elif self.fusion_mode == 'concat_attn':
            # 简单拼接特征
            concat_feat = torch.cat([spatial_feat, temporal_feat], dim=-1)  # [B, 2*embed_dim]
            concat_feat = self.norm1(concat_feat)
            # 投影到embed_dim并做注意力
            proj_feat = self.proj(concat_feat).unsqueeze(1)  # [B, 1, embed_dim]
            proj_feat = proj_feat + self.attention(proj_feat)
            proj_feat = proj_feat + self.mlp(self.norm2(proj_feat))
            out = proj_feat.squeeze(1)  # [B, embed_dim]
        
        # 输出层
        out = self.output_layer(out)
        return out


class FCN(nn.Module):
        def __init__(self, n_in, n_out, n_ffeatures, n_hidden, n_layers):
            super().__init__()
            self.pi = torch.acos(torch.zeros(1)).item() * 2
            self.n_ffeatures = n_ffeatures
            if n_ffeatures != 0:
                self.B = torch.nn.Parameter(torch.randn((n_in, n_ffeatures)))
                self.b = torch.nn.Parameter(torch.randn((1, n_ffeatures)))
                n_in = n_ffeatures

            activation = nn.Tanh
            self.fcs = nn.Sequential(*[
                            nn.Linear(n_in, n_hidden),
                            activation()])
            self.fch = nn.Sequential(*[
                            nn.Sequential(*[
                                nn.Linear(n_hidden, n_hidden),
                                activation()]) for _ in range(n_layers-1)])
            self.fce = nn.Linear(n_hidden, n_out)

        def forward(self, r):
            if self.n_ffeatures != 0:
                # ( r @ B ) * (2*pi)
                r = torch.cos( 2*self.pi*r @ self.B + self.b ) # Fourier mapping
            r = self.fcs(r)
            r = self.fch(r)
            r = self.fce(r)
            return r

class FCN_with_Attention(nn.Module):
    """
    FCN主干 + Attention增强
    在原FCN结构基础上，添加自注意力残差模块作为增强
    
    通过将隐藏特征拆分为多个token，使Attention能捕捉特征间的依赖关系
    可通过use_attention=False退化为原FCN

    Args:
        n_in: 输入维度 (3表示x,y,t)
        n_out: 输出维度
        n_ffeatures: 傅里叶特征维度 (0表示不使用)
        n_hidden: 隐藏层维度 (需被n_tokens整除)
        n_layers: FCN隐藏层数量
        num_heads: 注意力头数量
        n_tokens: 拆分的token数量
        mlp_ratio: Attention后MLP的中间层维度比例
        drop_ratio: Dropout比例
        use_attention: 是否使用Attention增强 (False时退化为原FCN)

        n_hidden % n_tokens == 0 且 (n_hidden / n_tokens) % num_heads == 0
    """
    def __init__(self, n_in=3, n_out=1, n_ffeatures=64, n_hidden=64, n_layers=3,
                 num_heads=4, n_tokens=4, mlp_ratio=4., drop_ratio=0., use_attention=True):
        super().__init__()
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        self.n_ffeatures = n_ffeatures
        self.use_attention = use_attention
        self.n_tokens = n_tokens
        
        # 傅里叶特征映射 (与原FCN相同)
        if n_ffeatures != 0:
            self.B = nn.Parameter(torch.randn((n_in, n_ffeatures)))
            self.b = nn.Parameter(torch.randn((1, n_ffeatures)))
            encoder_in = n_ffeatures
        else:
            encoder_in = n_in
        
        # FCN主干 (与原FCN结构完全一致)
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
            nn.Linear(encoder_in, n_hidden),
            activation()
        ])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(n_hidden, n_hidden),
                activation()
            ]) for _ in range(n_layers - 1)
        ])
        
        # ========== Attention增强模块 ==========
        if use_attention:
            assert n_hidden % n_tokens == 0, "n_hidden must be divisible by n_tokens"
            token_dim = n_hidden // n_tokens  # 每个token的维度
            
            self.attn_norm1 = nn.LayerNorm(token_dim)
            self.attention = Attention(
                dim=token_dim, 
                num_heads=num_heads, 
                attn_drop=drop_ratio, 
                proj_drop=drop_ratio
            )
            self.attn_norm2 = nn.LayerNorm(token_dim)
            self.attn_mlp = MLP(
                in_features=token_dim, 
                hidden_features=int(token_dim * mlp_ratio), 
                out_features=token_dim,
                act_layer=nn.GELU,
                drop=drop_ratio
            )
        # =============================================
        
        # 输出层
        self.fce = nn.Linear(n_hidden, n_out)
    
    def forward(self, r):
        # 傅里叶特征映射
        if self.n_ffeatures != 0:
            r = torch.cos(2 * self.pi * r @ self.B + self.b)
        
        # FCN主干编码
        r = self.fcs(r)
        r = self.fch(r)  # [B, n_hidden]
        
        # Attention增强 (残差连接)
        if self.use_attention:
            batch = r.shape[0]
            # 拆分为多个token: [B, n_hidden] -> [B, n_tokens, token_dim]
            h = r.reshape(batch, self.n_tokens, -1)
            # 自注意力 + 残差 (Pre-Norm)
            h = h + self.attention(self.attn_norm1(h))
            # MLP + 残差
            h = h + self.attn_mlp(self.attn_norm2(h))
            # 还原形状: [B, n_tokens, token_dim] -> [B, n_hidden]
            r = h.reshape(batch, -1)
        
        # 输出层
        r = self.fce(r)
        return r


class LongHistoryTemporalEncoderPINN(nn.Module):
    """
    Long-History Temporal Encoder PINN (用于 2D wave equation)
    Note:
        This module does not depend on causal training schedule.
        Under standard training, it still encodes temporal context by
        constructing a fixed-length past-time window for each current t.

    结构：
        (x, y, t)
        ├── 空间分支 (x, y)
        │   → 可选 Fourier feature mapping
        │   → MLP: Linear → Tanh → Linear
        │   → s_ctx: [B, d_h]
        │
        └── 时间分支 (t)
            → _build_past_time_window: [B, 1] → [B, L, 1]
            → temporal_embed: Linear(1 → d_t)  → X_t: [B, L, d_t]
            → LSTM(d_t → d_h, batch_first=True, bidirectional=False)
            → t_ctx = H[:, -1, :]  [B, d_h]

        融合：z = cat([s_ctx, t_ctx], dim=-1)  [B, 2*d_h]
        输出头：LayerNorm → Linear(2d_h→d_h) → Tanh → Linear(d_h→1)

    Args:
        seq_len             : 时间历史序列长度 L
        dt_hist             : 时间历史步长 dt_hist
        temporal_embed_dim  : 时间 embedding 维度 d_t
        hidden_dim          : LSTM 隐藏维度 d_h（空间分支输出同维度）
        spatial_hidden      : 空间 MLP 中间层维度
        spatial_layers      : 空间 MLP 隐藏层数量（≥1）
        n_ffeatures_spatial : 空间 Fourier 特征数（0=不使用）
        use_fourier_temporal: 是否对时间序列也做 Fourier（默认 False）
        n_ffeatures_temporal: 时间 Fourier 特征数（use_fourier_temporal=True 时生效）

    硬约束：
        - 禁止 attention / multihead attention / transformer
        - 禁止 mean pooling / decoder LSTM / cross attention
        - t_ctx 必须取 H[:, -1, :]
        - 支持二阶 autograd（不 detach，不 numpy）
    """

    def __init__(
        self,
        seq_len: int = 8,
        dt_hist: float = 0.02,
        temporal_embed_dim: int = 16,
        hidden_dim: int = 32,
        spatial_hidden: int = 32,
        spatial_layers: int = 2,
        n_ffeatures_spatial: int = 64,
        use_fourier_temporal: bool = False,
        n_ffeatures_temporal: int = 16,
    ):
        super().__init__()

        self.seq_len = seq_len                          # L
        self.dt_hist = dt_hist                          # 历史步长
        self.d_t = temporal_embed_dim                   # 时间 embedding 维度
        self.d_h = hidden_dim                           # LSTM 隐藏维度
        self.use_fourier_spatial = (n_ffeatures_spatial != 0)
        self.use_fourier_temporal = use_fourier_temporal
        self.n_ffeatures_spatial = n_ffeatures_spatial
        self.n_ffeatures_temporal = n_ffeatures_temporal
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        # ---- 预计算时间偏移 buffer: [1, L, 1] ----
        # [-(L-1)*dt, -(L-2)*dt, ..., -dt, 0]
        offsets = -torch.arange(seq_len - 1, -1, -1, dtype=torch.float32) * dt_hist
        self.register_buffer("time_offsets", offsets.view(1, seq_len, 1))
        # time_offsets: [1, L, 1]

        # ============================================================
        # 1. 空间分支
        # ============================================================
        if self.use_fourier_spatial:
            # 空间 Fourier 参数: cos(2π · x @ B_s + b_s)
            self.B_spatial = nn.Parameter(torch.randn(2, n_ffeatures_spatial))
            self.b_spatial = nn.Parameter(torch.randn(1, n_ffeatures_spatial))
            spatial_in = n_ffeatures_spatial
        else:
            spatial_in = 2  # (x, y) 原始输入

        # 空间 MLP: Linear → Tanh → [hidden] × (layers-1) → Linear(d_h)
        # 结构: Linear(spatial_in → spatial_hidden) → Tanh
        #        × (spatial_layers - 1)
        #       Linear(spatial_hidden → d_h)
        spatial_layers = max(1, spatial_layers)
        layers_s = [nn.Linear(spatial_in, spatial_hidden), nn.Tanh()]
        for _ in range(spatial_layers - 1):
            layers_s += [nn.Linear(spatial_hidden, spatial_hidden), nn.Tanh()]
        layers_s.append(nn.Linear(spatial_hidden, hidden_dim))
        self.spatial_mlp = nn.Sequential(*layers_s)
        # 输出: s_ctx [B, d_h]

        # ============================================================
        # 2. 时间分支
        # ============================================================
        if use_fourier_temporal:
            # 对时间序列每一步做 Fourier: cos(2π · t @ B_t + b_t)
            self.B_temporal = nn.Parameter(torch.randn(1, n_ffeatures_temporal))
            self.b_temporal = nn.Parameter(torch.randn(1, n_ffeatures_temporal))
            lstm_input_dim = n_ffeatures_temporal
        else:
            lstm_input_dim = 1  # 原始时间标量

        # 时间 embedding: Linear(1 → d_t) 或 Linear(n_ffeatures_temporal → d_t)
        self.temporal_embed = nn.Linear(lstm_input_dim, temporal_embed_dim)
        # 输出: X_t [B, L, d_t]

        # LSTM encoder: 单向，单层，batch_first=True
        self.lstm = nn.LSTM(
            input_size=temporal_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        # 输出: H [B, L, d_h]，取 t_ctx = H[:, -1, :] [B, d_h]

        # ============================================================
        # 3. 输出头: LayerNorm → Linear(2d_h→d_h) → Tanh → Linear(d_h→1)
        # ============================================================
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_past_time_window(self, t: torch.Tensor) -> torch.Tensor:
        """
        Build past-time history for temporal context encoding.
        This function is for sequence construction in the LSTM branch,
        not for controlling training curriculum.

        Args:
            t: [B, 1] 当前时间点
        Returns:
            t_seq: [B, L, 1]
                   序列定义: [t-(L-1)*dt_hist, t-(L-2)*dt_hist, ..., t-dt_hist, t]
        """
        # t: [B, 1] → [B, 1, 1]
        # time_offsets: [1, L, 1]
        t_seq = t.unsqueeze(1) + self.time_offsets   # [B, L, 1]
        t_seq = torch.clamp(t_seq, min=0.0)           # 截断负时间，物理约束
        return t_seq                                   # [B, L, 1]

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: [B, 3]  列含义: r[:,0]=x, r[:,1]=y, r[:,2]=t
        Returns:
            out: [B, 1]  波场预测值
        """
        B = r.shape[0]

        # ---- Step 1: 分离空间和时间 ----
        r_s = r[:, :2]      # [B, 2]  (x, y)
        t   = r[:, 2:3]     # [B, 1]  (t)

        # ---- Step 2: 空间分支 ----
        if self.use_fourier_spatial:
            # Fourier mapping: cos(2π · r_s @ B_spatial + b_spatial)
            phi_s = torch.cos(2 * self.pi * r_s @ self.B_spatial + self.b_spatial)
            # phi_s: [B, n_ffeatures_spatial]
        else:
            phi_s = r_s     # [B, 2]

        s_ctx = self.spatial_mlp(phi_s)     # [B, d_h]

        # ---- Step 3: 时间分支 ----
        # Step 3.1: 构造历史时间窗口
        t_seq = self._build_past_time_window(t)     # [B, L, 1]

        # Step 3.2: 可选 Fourier feature（逐时间步处理）
        if self.use_fourier_temporal:
            # t_seq: [B, L, 1] → reshape → [B*L, 1]
            t_flat = t_seq.reshape(B * self.seq_len, 1)     # [B*L, 1]
            phi_t_flat = torch.cos(
                2 * self.pi * t_flat @ self.B_temporal + self.b_temporal
            )                                               # [B*L, n_ffeatures_temporal]
            t_seq_in = phi_t_flat.reshape(B, self.seq_len, -1)  # [B, L, n_ffeatures_temporal]
        else:
            t_seq_in = t_seq    # [B, L, 1]

        # Step 3.3: 时间 embedding
        # t_seq_in: [B, L, lstm_input_dim] → reshape → [B*L, lstm_input_dim]
        t_in_flat = t_seq_in.reshape(B * self.seq_len, -1)         # [B*L, lstm_input_dim]
        X_t_flat  = self.temporal_embed(t_in_flat)                  # [B*L, d_t]
        X_t       = X_t_flat.reshape(B, self.seq_len, self.d_t)     # [B, L, d_t]

        # Step 3.4: LSTM encoder
        H, _ = self.lstm(X_t)           # H: [B, L, d_h]

        # Step 3.5: 取最后时间步作为时间 context（因果，不做 pooling 或 attention）
        t_ctx = H[:, -1, :]             # [B, d_h]

        # ---- Step 4: 融合 ----
        z = torch.cat([s_ctx, t_ctx], dim=-1)   # [B, 2*d_h]

        # ---- Step 5: 输出 ----
        out = self.head(z)              # [B, 1]

        return out
    


