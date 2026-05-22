# `main.py` 实验参数汇总（排除 `train_st`）

本文整理 `main.py` 运行过程中除 `train_st` 以外各阶段的模型结构参数与实验参数。涉及阶段：

```text
1. train_kan_st
2. train_kan
4. finetune
5. evaluate
6. visualize
```

不包含 `main.py` 的第 3 步 `train_st` / `train_st_pinn.py --ablation`。

## 1. `main.py` 调度关系

`main.py` 的默认流水线是：

```python
DEFAULT_PIPELINE = ["train_kan_st", "train_kan", "train_st", "finetune", "evaluate", "visualize"]
```

本文件只讨论其中除 `train_st` 外的步骤。

| main 阶段 | 函数 | 实际命令 | 涉及模型 | 涉及数据集 |
|---|---|---|---|---|
| 1. `train_kan_st` | `step_train_kan_st()` | `python train_kan_pinn.py --dataset <ds> --epochs <epochs>` | KAN-ST-PINN | `threelayer`、`marmousi`、`overthrust` |
| 2. `train_kan` | `step_train_kan_pure()` | `python train_kan_pinn.py --dataset <ds> --pure-kan --epochs <epochs>` | Pure-KAN / KAN-PINN | 同上 |
| 4. `finetune` | `step_finetune()` | `python finetune.py --dataset <ds> ...` 和 `python finetune.py --dataset <ds> --pure-kan ...` | KAN-ST-PINN R6、Pure-KAN R6 | 同上 |
| 5. `evaluate` | `step_evaluate()` | `python compute_l2.py`、`python evaluate.py` | R5 主线模型、R6 微调模型 | 同上 |
| 6. `visualize` | `step_visualize()` | `python vis_compare.py`、`python plot_abl.py` | R5 KAN-ST-PINN、R5 Pure-KAN | 同上 |

`main.py` 的全局参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--epochs` | `10000` | 传给阶段 1、2 的 `train_kan_pinn.py`；`train_st` 也会用，但本文排除 |
| `--finetune-epochs` | `4000` | 传给阶段 4 的 `finetune.py` |
| `--quick` | off | 在 `main.py` 层把 `epochs=600`、`finetune_epochs=200`；注意这不是 `train_kan_pinn.py --quick` |
| `--eval-only` | off | 只运行 `evaluate` + `visualize` |
| `--finetune-only` | off | 只运行 `finetune` + `evaluate` + `visualize` |
| `--resume-step` | `None` | 按默认流水线编号续跑；与 `--steps` / `--eval-only` / `--finetune-only` 互斥 |

## 2. 数据集参数

三个数据集在 `train_kan_pinn.py` 与 `PINNs_util/datasets.py` 中生成。阶段 1、2、4 使用 `train_kan_pinn.prepare_dataset(config, device)`；阶段 5、6 的 `compute_l2.py` / `vis_compare.py` / `plot_abl.py` 也按同一套数据配置重建数据。

| 数据集 | 生成函数 | 物理区域 | 物理时长 | 速度归一化 | 默认网格 | 初始脉冲 | 归一化计算时长 |
|---|---|---:|---:|---|---:|---|---:|
| `threelayer` | `train_kan_pinn.prepare_threelayer()` | `L=5.0 km` | `T=2.0 s` | `c0=3.0 km/s` | `Nx=Ny=60` | 中心 `(0.5, 0.5)`，`std=0.05` | `T_norm = T*c0/L = 1.2` |
| `marmousi` | `generate_marmousi()` | `L=3.0 km` | `T=1.0 s` | `c_max=4.5 km/s` | `Nx=Ny=60` | 中心 `(0.5, 0.15)`，`std=0.04` | `T_norm = T_s*c_max/L = 1.5` |
| `overthrust` | `generate_overthrust()` | `L=4.0 km` | `T=0.8 s` | `c_max=6.0 km/s` | `Nx=Ny=60` | 中心 `(0.5, 0.12)`，`std=0.035` | `T_norm = T_s*c_max/L = 1.2` |

说明：

- `threelayer` 的速度场为三层 sigmoid 速度模型，核心形式为 `0.5 + 0.25*sigmoid(y-0.33) + 0.25*sigmoid(y-0.66)`，`steepness=40.0`。
- `marmousi` 是合成 Marmousi 风格速度场，物理速度范围约 `1.5-4.5 km/s`。
- `overthrust` 是合成 Overthrust 风格速度场，物理速度范围约 `2.0-6.0 km/s`。
- 三个数据集都会先用有限差分求参考波场 `p_ref`，再按最大绝对值归一化。

## 3. 模型结构参数

### 3.1 KAN-ST-PINN

用于：

- 阶段 1：`train_kan_st`
- 阶段 4：`finetune` 中的 KAN-ST-PINN R6
- 阶段 5：`compute_l2.py` / `evaluate.py` 重建模型做评估
- 阶段 6：`vis_compare.py` / `plot_abl.py` 重建模型做可视化

结构来源：`train_kan_pinn.Config` + `PINNs_util/kan_pinn.py::KANSTPinn`。

| 参数 | 值 | 含义 |
|---|---:|---|
| `d_model` | `128` | 主隐藏维度 |
| `n_fourier` | `32` | Fourier 特征频率数 |
| `spatial_kan_layers` | `3` | 空间 KAN 编码器层数 |
| `grid_size` | `5` | B-spline 网格区间数 |
| `spline_order` | `3` | 三次 B-spline |
| `seq_len` | `8` | 时间窗口长度；训练/评估时会按数据时间步做上限对齐 |
| `dt_hist` | `0.02` 初始值 | 时间历史间隔；KAN-ST 会在 `align_temporal_config()` / `align_temporal()` 中对齐到数据 `dt` |
| `lstm_layers` | `2` | BiLSTM 层数 |
| `num_heads` | `4` | Cross-attention 头数 |
| `n_attn_layers` | `2` | Cross-attention + FFN 层数 |
| `dropout` | `0.0` | dropout |

结构路径：

```text
(x, y, t)
-> 空间 Fourier 特征：2*n_fourier + 2 = 66 维
-> 空间 KAN 编码器
-> 时间 Fourier 特征：2*n_fourier + 1 = 65 维
-> BiLSTM 时间编码器
-> Cross-Attention 融合
-> KAN 输出头
-> p(x,y,t)
```

运行日志中默认参数量为 `985,376`：

```text
kan(spline+base): 494,720
fourier: 160
lstm: 215,168
attention+ffn: 264,960
other: 10,368
```

### 3.2 Pure-KAN / KAN-PINN

用于：

- 阶段 2：`train_kan`
- 阶段 4：`finetune` 中的 Pure-KAN R6
- 阶段 5：`compute_l2.py` / `evaluate.py` 重建模型做评估
- 阶段 6：`vis_compare.py` / `plot_abl.py` 重建模型做可视化

结构来源：`train_kan_pinn.Config(pure_kan=True)` + `PINNs_util/kan_pinn.py::KANPinn`。

| 参数 | 值 | 含义 |
|---|---:|---|
| `d_hidden` / `d_model` | `128` | KAN 隐藏维度 |
| `n_layers` / `spatial_kan_layers` | `3` | KAN 隐藏层数 |
| `n_fourier` | `32` | Fourier 特征频率数 |
| `grid_size` | `5` | B-spline 网格区间数 |
| `spline_order` | `3` | 三次 B-spline |
| `use_fourier` | `True` | 使用空间和时间 Fourier 特征 |
| `use_layernorm` | `True` | 每层 KAN 后使用 LayerNorm |
| LSTM / Attention | 无 | `--pure-kan` 下不使用时序 LSTM 与 Attention |

输入特征维度：

```text
4*n_fourier + 3 = 131
```

即空间 `sin/cos`、时间 `sin/cos` 加原始 `(x,y,t)`。

运行日志中默认参数量为 `497,568`：

```text
spline: 446,976
base_linear: 49,664
fourier: 160
other: 768
```

## 4. 阶段 1：`train_kan_st`

命令：

```bash
python train_kan_pinn.py --dataset <dataset> --epochs <args.epochs>
```

对三个数据集逐一运行。

| 项目 | KAN-ST-PINN 设置 |
|---|---|
| 模型 | `KANSTPinn` |
| `pure_kan` | `False` |
| 数据集 | `threelayer`、`marmousi`、`overthrust` |
| epoch | 默认 `10000`；`main.py --quick` 时为 `600`；也可由 `main.py --epochs` 覆盖 |
| 学习率 | `1e-3` |
| 优化器 | Adam |
| LR warmup | `warmup_epochs=500` |
| LR 最小比例 | `lr_min_ratio=0.01` |
| 初始条件时间步数 | `n_ini=20` |
| 监督数据点 | `n_data=5000` |
| collocation 点 | `n_colloc=4000` |
| 边界点 | `n_bc=800` |
| collocation Gaussian 比例 | `ratio_gaussian=0.85` |
| 边界条件 | `bc_type='absorbing'` |
| 梯度裁剪 | `grad_clip=1.0` |
| 初始条件 mini-batch | `mini_batch_ini=4000` |
| 损失权重 | `w_ini=10.0`、`w_pde=0.01`、`w_bc=0.01`、`w_data=10.0` |
| 自适应 lambda | `use_adaptive_lambda=False` |
| 因果加权 | `use_causal=True`，`causal_eps: 3.0 -> 0.5` |
| 晚期时间监督加权 | `late_data_alpha=3.0`，当前训练代码实际使用 `1 + alpha*(t/T)^2` |
| 晚期 collocation 覆盖 | `late_colloc_uniform=0.5` |
| PDE warmup / rampup | `pde_warmup_epochs=2000`、`pde_rampup_epochs=3000` |
| collocation / boundary pool | `pool_size=100`，每 100 epoch 循环使用预生成池 |
| 随机种子 | `set_seed(0)` |
| 输出模型目录 | `trained/<dataset>/kan_st_pinn/` |
| 输出图片目录 | `figures/<dataset>/kan_st_pinn/` |

额外注意：

- KAN-ST 会在建模前执行 `align_temporal_config(config, data)`，把 `dt_hist` 对齐到数据的实际时间步，并在必要时缩小 `seq_len`。
- 每 500 epoch 保存一次 checkpoint：`checkpoint_epoch*.pt`，最终保存 `model.pt` 和 `history.pt`。

## 5. 阶段 2：`train_kan`

命令：

```bash
python train_kan_pinn.py --dataset <dataset> --pure-kan --epochs <args.epochs>
```

对三个数据集逐一运行。

| 项目 | Pure-KAN 设置 |
|---|---|
| 模型 | `KANPinn` |
| `pure_kan` | `True` |
| 数据集 | `threelayer`、`marmousi`、`overthrust` |
| epoch | 默认 `10000`；`main.py --quick` 时为 `600`；也可由 `main.py --epochs` 覆盖 |
| 学习率 | `1e-3` |
| 监督数据点 | `n_data=5000` |
| collocation 点 | `n_colloc=4000` |
| 边界点 | `n_bc=800` |
| 损失权重 | `w_ini=10.0`、`w_pde=0.01`、`w_bc=0.01`、`w_data=10.0` |
| 因果加权 | `use_causal=True`，`causal_eps: 3.0 -> 0.5` |
| PDE warmup / rampup | `2000 / 3000` |
| 输出模型目录 | `trained/<dataset>/kan_pinn/` |
| 输出图片目录 | `figures/<dataset>/kan_pinn/` |

除模型结构外，Pure-KAN 与阶段 1 的训练超参基本一致。区别是 `pure_kan=True` 后不会构建 LSTM/Attention，也不会执行 `align_temporal_config()`，因为 Pure-KAN 没有时间窗口模块。

## 6. 阶段 4：`finetune` / R6 微调

`main.py` 对每个数据集跑两次：先 KAN-ST-PINN，再 Pure-KAN。

### 6.1 KAN-ST-PINN R6

命令：

```bash
python finetune.py --dataset <dataset> --epochs <args.finetune_epochs> --n-data 5000 --n-colloc 4500
```

| 项目 | 设置 |
|---|---|
| 源模型 | `trained/<dataset>/kan_st_pinn/model.pt` 或最新 `checkpoint_epoch*.pt` |
| 输出模型 | `trained/<dataset>/kan_st_pinn_r6/model.pt` |
| 输出图片 | `figures/<dataset>/kan_st_pinn_r6/` |
| 模型结构 | 继承 KAN-ST-PINN 结构参数 |
| epoch | 默认 `4000`；`main.py --quick` 时为 `200`；可由 `--finetune-epochs` 覆盖 |
| 学习率 | `2e-4` |
| LR warmup | `200` |
| LR 最小比例 | `0.05`，即最低约 `1e-5` |
| 监督数据点 | `5000`，这是 `main.py` 对 KAN-ST 的覆盖值 |
| collocation 点 | `4500`，这是 `main.py` 对 KAN-ST 的覆盖值 |
| 边界点 | `n_bc=1200`，来自 `R6Config` |
| PDE warmup / rampup | `0 / 0`，R6 直接启用 PDE |
| 晚期监督权重 | `late_data_alpha=10.0` |
| 晚期 collocation 覆盖 | `late_colloc_uniform=0.7` |
| 因果加权 | `causal_eps: 1.0 -> 0.3` |
| 随机种子 | `set_seed(0)` |

注意：`R6Config` 定义了 `late_data_power=3`，但当前 `train_kan_pinn.train_model()` 里实际写死使用 `t_frac ** 2`，没有读取 `late_data_power`。因此按当前源码，R6 晚期监督仍是二次幂时间权重，只是 `alpha` 从 `3.0` 提升到 `10.0`。

### 6.2 Pure-KAN R6

命令：

```bash
python finetune.py --dataset <dataset> --epochs <args.finetune_epochs> --n-data 8000 --n-colloc 6000 --pure-kan
```

| 项目 | 设置 |
|---|---|
| 源模型 | `trained/<dataset>/kan_pinn/model.pt` 或最新 `checkpoint_epoch*.pt` |
| 输出模型 | `trained/<dataset>/kan_pinn_r6/model.pt` |
| 输出图片 | `figures/<dataset>/kan_pinn_r6/` |
| 模型结构 | 继承 Pure-KAN 结构参数 |
| epoch | 默认 `4000`；`main.py --quick` 时为 `200` |
| 学习率 | `2e-4` |
| 监督数据点 | `8000` |
| collocation 点 | `6000` |
| 边界点 | `1200` |
| PDE warmup / rampup | `0 / 0` |
| 晚期监督权重 | `late_data_alpha=10.0`，当前代码实际二次幂加权 |
| 晚期 collocation 覆盖 | `late_colloc_uniform=0.7` |
| 因果加权 | `causal_eps: 1.0 -> 0.3` |

Pure-KAN R6 与 KAN-ST-PINN R6 的主要实验差异是 `n_data/n_colloc` 更大，并且模型没有 LSTM/Attention。

## 7. 阶段 5：`evaluate`

阶段 5 不训练模型。它包含两部分。

### 7.1 `compute_l2.py`

用途：评估 R5 / 主线模型，即：

```text
kan_st_pinn/threelayer
kan_st_pinn/marmousi
kan_st_pinn/overthrust
kan_pinn/threelayer
kan_pinn/marmousi
kan_pinn/overthrust
```

结构参数在 `compute_l2.py` 中硬编码，与训练默认值一致：

| 参数 | 值 |
|---|---:|
| `D_MODEL` | `128` |
| `N_FOURIER` | `32` |
| `SPATIAL_KAN_LAYERS` | `3` |
| `GRID_SIZE` | `5` |
| `SPLINE_ORDER` | `3` |
| `SEQ_LEN` | `8` |
| `DT_HIST` | `0.02` |
| `LSTM_LAYERS` | `2` |
| `NUM_HEADS` | `4` |
| `N_ATTN_LAYERS` | `2` |
| `DROPOUT` | `0.0` |
| `NX=NY` | `60` |

评估设置：

| 项目 | 设置 |
|---|---|
| 权重加载 | 优先 `model.pt`，否则最新 `checkpoint_epoch*.pt` |
| 预测 batch size | `4096` |
| 指标 | 全局 `Rel L2`、`MAE`、逐时刻 `Rel L2` 的 mean/max |
| 旧基线 | 只对 `threelayer` 使用 `Old ST-PINN = 0.113` |
| 输出 | 终端表格，不直接写 `figures/` |

### 7.2 `evaluate.py`

用途：评估 R6 微调模型，并与写死的 R5 baseline 比较。

评估组合：

```text
kan_st_pinn_r6/<dataset>
kan_pinn_r6/<dataset>
```

R5 baseline：

| 模型/数据集 | R5 baseline |
|---|---:|
| `kan_st_pinn/threelayer` | `0.027574` |
| `kan_st_pinn/marmousi` | `0.041026` |
| `kan_st_pinn/overthrust` | `0.037547` |
| `kan_pinn/threelayer` | `0.098174` |
| `kan_pinn/marmousi` | `0.118788` |
| `kan_pinn/overthrust` | `0.124966` |

判定规则：

| 条件 | verdict |
|---|---|
| `R6 < 0.95 * R5` | `PASS` |
| `R6 < 1.05 * R5` | `MARGINAL` |
| 其它 | `FAIL` |

输出：

```text
r6_results.json
```

## 8. 阶段 6：`visualize`

阶段 6 不训练模型，但会重建模型、加载权重、计算指标并出图。

### 8.1 `vis_compare.py`

用途：生成论文风格汇总图，输出到：

```text
figures/paper/
```

模型结构参数硬编码为与 `compute_l2.py` 相同的默认值：`D_MODEL=128`、`N_FOURIER=32`、`GRID_SIZE=5`、`SPLINE_ORDER=3`、`SEQ_LEN=8`、`LSTM_LAYERS=2`、`NUM_HEADS=4`、`N_ATTN_LAYERS=2`。

评估组合：

| 模型 | 数据集 | 用途 |
|---|---|---|
| KAN-ST-PINN | 三个数据集 | 计算指标、逐时刻误差；生成 `fig2_wavefield_<dataset>.png` |
| Pure-KAN | 三个数据集 | 计算指标、逐时刻误差；参与柱状图和逐时刻曲线 |
| Old ST-PINN | 仅 `threelayer` 历史 | 参与 `fig1_loss_curves.png` 和旧基线对照 |

固定图表参数：

| 项目 | 设置 |
|---|---|
| 输出目录 | `figures/paper` |
| 旧基线 | `OLD_REL_L2 = 0.113` |
| `fig1` 历史 | `trained/threelayer/full/history_6000.pt`，不存在则用 `history.pt` |
| 波场图时刻 | `[0, n_T//4, n_T//2, 3*n_T//4, n_T-1]` |
| 预测 batch size | `4096` |

### 8.2 `plot_abl.py`

用途：生成 KAN-ST-PINN vs Pure-KAN 的拼图式对比图，输出到：

```text
figures/ablation/
```

评估组合：

| 模型 | 数据集 | 权重 |
|---|---|---|
| KAN-ST-PINN | 三个数据集 | `trained/<dataset>/kan_st_pinn/model.pt` 或最新 checkpoint |
| Pure-KAN | 三个数据集 | `trained/<dataset>/kan_pinn/model.pt` 或最新 checkpoint |

固定图表/实验参数：

| 项目 | 设置 |
|---|---|
| 输出目录 | `figures/ablation` |
| 预测 batch size | `4096` |
| 选取时刻 | `[0, n_T//4, n_T//2, 3*n_T//4, n_T-1]` |
| 指标 | 全局 `Rel L2`，以及选取时刻的 per-timestep `Rel L2` |
| 输出图片 | `ablation_<dataset>.png` |
| 输出文本 | `ablation_metrics.txt` |

注意：`plot_abl.py` 只比较 R5 主线 `kan_st_pinn` 与 `kan_pinn`，不读取 `*_r6` 目录。

## 9. 按模型和数据集的组合总表

| 阶段 | 模型 | 数据集 | 结构参数 | 主要实验参数 | 输出 |
|---|---|---|---|---|---|
| `train_kan_st` | KAN-ST-PINN | 3 个数据集 | `d_model=128`、`KAN layers=3`、`grid=5`、`spline=3`、`seq_len=8`、`BiLSTM=2`、`heads=4`、`attn_layers=2` | `epochs=10000`、`lr=1e-3`、`n_data=5000`、`n_colloc=4000`、`n_bc=800`、PDE `2000/3000` | `trained/<ds>/kan_st_pinn/`、`figures/<ds>/kan_st_pinn/` |
| `train_kan` | Pure-KAN | 3 个数据集 | `d_hidden=128`、`KAN layers=3`、`grid=5`、`spline=3`、`n_fourier=32`，无 LSTM/Attention | 同上 | `trained/<ds>/kan_pinn/`、`figures/<ds>/kan_pinn/` |
| `finetune` | KAN-ST-PINN R6 | 3 个数据集 | 同 KAN-ST-PINN | `epochs=4000`、`lr=2e-4`、`n_data=5000`、`n_colloc=4500`、`n_bc=1200`、PDE `0/0`、`late_alpha=10` | `trained/<ds>/kan_st_pinn_r6/`、`figures/<ds>/kan_st_pinn_r6/` |
| `finetune` | Pure-KAN R6 | 3 个数据集 | 同 Pure-KAN | `epochs=4000`、`lr=2e-4`、`n_data=8000`、`n_colloc=6000`、`n_bc=1200`、PDE `0/0`、`late_alpha=10` | `trained/<ds>/kan_pinn_r6/`、`figures/<ds>/kan_pinn_r6/` |
| `evaluate` | KAN-ST-PINN / Pure-KAN | 3 个数据集 | 评估脚本按默认结构重建 | batch `4096`，计算全局/逐时刻 `Rel L2`、`MAE` | 终端表格、`r6_results.json` |
| `visualize` | KAN-ST-PINN / Pure-KAN | 3 个数据集 | 可视化脚本按默认结构重建 | batch `4096`，选取 5 个时刻出图 | `figures/paper/`、`figures/ablation/` |

## 10. 需要注意的源码细节

1. `main.py --quick` 只改 `main.py` 传下去的 epoch 数：训练 `600`、微调 `200`。它不会触发 `train_kan_pinn.py` 自己的 `--quick` 分支，因此不会自动把 `Nx/Ny` 改成 `40`、`d_model` 改成 `96`。
2. `R6Config.late_data_power=3` 当前没有被训练循环使用；训练循环里晚期监督权重仍写成 `t_frac ** 2`。
3. 阶段 5 和阶段 6 会重新构造模型，因此这些脚本里的结构常量必须与训练时一致，否则加载权重会失败或指标不可信。
4. `plot_abl.py` 使用的是 R5 目录 `kan_st_pinn` / `kan_pinn`，不是 R6 目录。
