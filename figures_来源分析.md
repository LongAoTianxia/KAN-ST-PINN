# `figures/` 图片来源分析

本文按 `main.py` 的流水线阶段，说明当前 `figures/` 目录中各类图片分别由哪个阶段、哪个脚本和哪个函数生成。行号基于当前工作区文件。

## 1. `main.py` 的阶段顺序

`main.py` 中的默认流水线定义在 `main.py:250`：

```text
1. train_kan_st
2. train_kan
3. train_st
4. finetune
5. evaluate
6. visualize
```

对应关系如下：

| 阶段 | `main.py` 函数 | 调用脚本 | 是否直接产图 |
|---|---|---|---|
| 1 | `step_train_kan_st()` (`main.py:154`) | `train_kan_pinn.py --dataset <ds>` | 是 |
| 2 | `step_train_kan_pure()` (`main.py:168`) | `train_kan_pinn.py --dataset <ds> --pure-kan` | 是 |
| 3 | `step_train_st_pinn()` (`main.py:182`) | `train_st_pinn.py --dataset <ds> --ablation` | 是 |
| 4 | `step_finetune()` (`main.py:194`) | `finetune.py --dataset <ds>` 和 `finetune.py --dataset <ds> --pure-kan` | 是 |
| 5 | `step_evaluate()` (`main.py:214`) | `compute_l2.py`、`evaluate.py` | 否，主要打印指标/写 `r6_results.json` |
| 6 | `step_visualize()` (`main.py:226`) | `vis_compare.py`、`plot_abl.py` | 是 |

`<ds>` 取值为 `threelayer`、`marmousi`、`overthrust`。

## 2. 训练阶段生成的图片

### 2.1 `figures/<dataset>/kan_st_pinn/`

当前图片：

```text
figures/threelayer/kan_st_pinn/loss_curves.png
figures/threelayer/kan_st_pinn/comparison.png
figures/marmousi/kan_st_pinn/loss_curves.png
figures/marmousi/kan_st_pinn/comparison.png
figures/overthrust/kan_st_pinn/loss_curves.png
figures/overthrust/kan_st_pinn/comparison.png
```

来源：

| 图片 | main 阶段 | 代码链路 | 含义 |
|---|---|---|---|
| `loss_curves.png` | 阶段 1：`train_kan_st` | `main.py:154` -> `train_kan_pinn.py` -> `run()` (`train_kan_pinn.py:625`) -> `save_loss_curves()` (`train_kan_pinn.py:525`) | KAN-ST-PINN 训练过程中各类 loss 的对数曲线 |
| `comparison.png` | 阶段 1：`train_kan_st` | `main.py:154` -> `train_kan_pinn.py` -> `run()` (`train_kan_pinn.py:625`) -> `save_comparison_plot()` (`train_kan_pinn.py:544`) | 参考波场、KAN-ST-PINN 预测波场、误差的快照对比 |

路径在 `train_kan_pinn.py:628` 组装为：

```python
fig_dir = os.path.join(config.FIGURES_DIR, config.DATASET, tag)
```

当 `pure_kan=False` 时，`tag = "kan_st_pinn"`。

### 2.2 `figures/<dataset>/kan_pinn/`

当前图片：

```text
figures/threelayer/kan_pinn/loss_curves.png
figures/threelayer/kan_pinn/comparison.png
figures/marmousi/kan_pinn/loss_curves.png
figures/marmousi/kan_pinn/comparison.png
figures/overthrust/kan_pinn/loss_curves.png
figures/overthrust/kan_pinn/comparison.png
```

来源：

| 图片 | main 阶段 | 代码链路 | 含义 |
|---|---|---|---|
| `loss_curves.png` | 阶段 2：`train_kan` | `main.py:168` -> `train_kan_pinn.py --pure-kan` -> `save_loss_curves()` (`train_kan_pinn.py:525`) | Pure-KAN / KAN-PINN 训练 loss 曲线 |
| `comparison.png` | 阶段 2：`train_kan` | `main.py:168` -> `train_kan_pinn.py --pure-kan` -> `save_comparison_plot()` (`train_kan_pinn.py:544`) | 参考波场、Pure-KAN 预测波场、误差的快照对比 |

当 `pure_kan=True` 时，`train_kan_pinn.py:626` 令 `tag = "kan_pinn"`，所以输出进入 `figures/<dataset>/kan_pinn/`。

### 2.3 `figures/threelayer/full/` 和 `figures/threelayer/no_fourier/`

当前图片：

```text
figures/threelayer/full/loss_curves.png
figures/threelayer/full/comparison.png
figures/threelayer/no_fourier/loss_curves.png
figures/threelayer/no_fourier/comparison.png
```

来源：

| 图片 | main 阶段 | 代码链路 | 含义 |
|---|---|---|---|
| `loss_curves.png` | 阶段 3：`train_st` | `main.py:182` -> `train_st_pinn.py --ablation` -> `run_ablation()` -> `run_single()` (`train_st_pinn.py:509`) -> `save_loss_curves()` (`train_st_pinn.py:402`) | ST-PINN 消融模型的训练 loss 曲线 |
| `comparison.png` | 阶段 3：`train_st` | `main.py:182` -> `train_st_pinn.py --ablation` -> `run_single()` (`train_st_pinn.py:509`) -> `save_comparison_plot()` (`train_st_pinn.py:421`) | ST-PINN 消融模型的参考/预测/误差对比图 |

`train_st_pinn.py:514` 将输出路径组装为：

```python
fig_dir = os.path.join(config.FIGURES_DIR, config.DATASET, variant)
```

理论上，阶段 3 会对 `ST_PINN_VARIANTS` 中的多个 variant 产图，包括 `full`、`no_fourier`、`no_lstm`、`no_attention`、`no_lstm_attn`、`self_attn`。当前 `figures/` 中只看到 `threelayer/full` 和 `threelayer/no_fourier` 两组图，结合 `nohup1.out`，本次 `train_st` 在 `threelayer/no_lstm` 附近失败过，因此后续 variant/数据集没有完整产出。

## 3. R6 微调阶段生成的图片

当前图片：

```text
figures/<dataset>/kan_st_pinn_r6/loss_curves.png
figures/<dataset>/kan_st_pinn_r6/comparison.png
figures/<dataset>/kan_pinn_r6/loss_curves.png
figures/<dataset>/kan_pinn_r6/comparison.png
```

其中 `<dataset>` 为 `threelayer`、`marmousi`、`overthrust`。

来源：

| 图片 | main 阶段 | 代码链路 | 含义 |
|---|---|---|---|
| `kan_st_pinn_r6/loss_curves.png` | 阶段 4：`finetune` | `main.py:194` -> `finetune.py` -> `save_loss_curves()` (`finetune.py:136`) | KAN-ST-PINN 从 R5 权重继续微调后的 R6 loss 曲线 |
| `kan_st_pinn_r6/comparison.png` | 阶段 4：`finetune` | `main.py:194` -> `finetune.py` -> `save_comparison_plot()` (`finetune.py:137`) | R6 KAN-ST-PINN 的参考/预测/误差对比 |
| `kan_pinn_r6/loss_curves.png` | 阶段 4：`finetune` | `main.py:194` -> `finetune.py --pure-kan` -> `save_loss_curves()` (`finetune.py:136`) | Pure-KAN 从 R5 权重继续微调后的 R6 loss 曲线 |
| `kan_pinn_r6/comparison.png` | 阶段 4：`finetune` | `main.py:194` -> `finetune.py --pure-kan` -> `save_comparison_plot()` (`finetune.py:137`) | R6 Pure-KAN 的参考/预测/误差对比 |

`finetune.py:71` 组装路径：

```python
fig_dir = os.path.join(config.FIGURES_DIR, config.DATASET, dst_tag)
```

`dst_tag` 为 `kan_st_pinn_r6` 或 `kan_pinn_r6`。

## 4. 评估阶段

`main.py` 的阶段 5 调用：

```text
compute_l2.py
evaluate.py
```

这一步不直接生成 `figures/` 下图片。`compute_l2.py` 主要计算并打印 R5/主线模型 Rel L2；`evaluate.py` 主要检查 R6 结果，并写出根目录下的 `r6_results.json`。

## 5. 可视化阶段生成的论文图

### 5.1 `figures/paper/`

来源：`main.py` 阶段 6 的第一部分，`step_visualize()` 调用 `vis_compare.py` (`main.py:226`、`main.py:229`)。

`vis_compare.py:47` 固定输出目录：

```python
FIG_DIR = "./figures/paper"
```

当前图片/文本：

| 文件 | 代码 | 含义 |
|---|---|---|
| `figures/paper/fig1_loss_curves.png` | `fig1_loss_curves()` (`vis_compare.py:134`)，保存于 `vis_compare.py:185` | three-layer 上 Old ST-PINN、KAN-ST-PINN、Pure-KAN 的多类 loss 曲线对比 |
| `figures/paper/fig2_wavefield_threelayer.png` | `fig2_wavefield()` (`vis_compare.py:192`)，保存于 `vis_compare.py:247`，由 `vis_compare.py:434` 调用 | threelayer 数据集的 KAN-ST-PINN 参考/预测/误差论文图 |
| `figures/paper/fig2_wavefield_marmousi.png` | 同上 | marmousi 数据集的 KAN-ST-PINN 参考/预测/误差论文图 |
| `figures/paper/fig2_wavefield_overthrust.png` | 同上 | overthrust 数据集的 KAN-ST-PINN 参考/预测/误差论文图 |
| `figures/paper/fig3_ablation_rel_l2.png` | `fig3_ablation_bars()` (`vis_compare.py:255`)，保存于 `vis_compare.py:300` | KAN-ST-PINN、Pure-KAN、Old ST-PINN 的全局 Rel L2 柱状对比 |
| `figures/paper/fig4_per_timestep_rel_l2.png` | `fig4_per_timestep()` (`vis_compare.py:307`)，保存于 `vis_compare.py:330` | 三个数据集上逐时刻 Rel L2 曲线 |
| `figures/paper/ablation_summary.txt` | `fig5_ablation_table()` (`vis_compare.py:337`)，保存于 `vis_compare.py:370` | 论文图对应的指标汇总文本，不是图片 |

### 5.2 `figures/ablation/`

来源：`main.py` 阶段 6 的第二部分，`step_visualize()` 调用 `plot_abl.py` (`main.py:226`、`main.py:233`)。

`plot_abl.py:21` 固定输出目录：

```python
OUT_DIR = "./figures/ablation"
```

当前图片/文本：

| 文件 | 代码 | 含义 |
|---|---|---|
| `figures/ablation/ablation_threelayer.png` | `make_ablation_figure()` (`plot_abl.py:67`)，保存于 `plot_abl.py:149` | threelayer 上 Reference、KAN-ST-PINN、Pure-KAN 与所选时刻误差柱状图 |
| `figures/ablation/ablation_marmousi.png` | 同上 | marmousi 上 Reference、KAN-ST-PINN、Pure-KAN 与所选时刻误差柱状图 |
| `figures/ablation/ablation_overthrust.png` | 同上 | overthrust 上 Reference、KAN-ST-PINN、Pure-KAN 与所选时刻误差柱状图 |
| `figures/ablation/ablation_metrics.txt` | `plot_abl.py:192` | 三个 ablation 图对应的 checkpoint 与误差指标文本，不是图片 |

## 6. 当前源码不能归到 `main.py` 的图片

以下图片在当前仓库源码中没有被 `main.py` 的默认流水线直接生成，或者生成脚本不在 `main.py` 调用列表里。这里按证据强弱标注。

### 6.1 `figures/threelayer/full/loss_analysis.png`

来源不是 `main.py`，而是辅助脚本 `hist_analysis.py`：

```text
hist_analysis.py:10  OUT_DIR = 'H:/wusheng/16/figures/threelayer/full'
hist_analysis.py:85  path = os.path.join(OUT_DIR, 'loss_analysis.png')
```

该脚本还会在 `hist_analysis.py:117` 调用 `train_st_pinn.save_comparison_plot()`，可能生成或覆盖 `full/comparison.png`。但 `full/loss_curves.png` 和 `full/comparison.png` 本身也能由 `main.py` 阶段 3 的 `train_st_pinn.py --ablation` 生成；`loss_analysis.png` 则只能在当前源码中追到 `hist_analysis.py`。

### 6.2 `figures/threelayer/full_T2s/`

当前图片：

```text
figures/threelayer/full_T2s/comparison_T2s.png
figures/threelayer/full_T2s/error_vs_time.png
```

来源不是 `main.py`，而是辅助脚本 `predict_kan_st_t25.py`。证据：

```text
predict_kan_st_t25.py:167  默认 out-dir = ./figures/threelayer/full_T2s
predict_kan_st_t25.py:203  comparison_path = ...
predict_kan_st_t25.py:204  error_path = ...
predict_kan_st_t25.py:207  save_comparison_figure(...)
predict_kan_st_t25.py:214  save_error_curve(...)
```

注意：脚本当前默认文件名是 `comparison_T2p5s.png` 和 `error_vs_time_T2p5s.png`，而现有文件名是 `comparison_T2s.png` 和 `error_vs_time.png`。因此它们应是运行该脚本时传入了 `--comparison-name` / `--error-name`，或来自脚本早期版本。

### 6.3 `figures/final/`

当前图片：

```text
figures/final/fig_kan_st_threelayer.png
figures/final/fig_kan_st_marmousi.png
figures/final/fig_kan_st_overthrust.png
```

当前源码中没有 `figures/final`、`fig_kan_st_` 的 `savefig` 或复制逻辑。`rg` 只能在说明文档里搜到 `figures/final`，不能在可执行 Python 脚本里搜到生成代码。

从命名看，这三张图很可能是 KAN-ST-PINN 的最终论文/汇报版波场图，语义上接近 `vis_compare.py` 的 `fig2_wavefield_<dataset>.png`，但尺寸和文件名都不同，所以不能严格认定为 `main.py` 阶段 6 的直接产物。结论：这是外部脚本、手工整理或未纳入当前仓库的绘图代码生成的图片。

### 6.4 `figures/paper_r6/`

当前图片：

```text
figures/paper_r6/r5_vs_r6_t2_threelayer_kan_st.png
figures/paper_r6/r5_vs_r6_t2_marmousi_kan_st.png
figures/paper_r6/r5_vs_r6_t2_overthrust_kan_st.png
figures/paper_r6/wavefield_kan_st_threelayer_r5.png
figures/paper_r6/wavefield_kan_st_threelayer_r6.png
figures/paper_r6/wavefield_kan_st_marmousi_r5.png
figures/paper_r6/wavefield_kan_st_marmousi_r6.png
figures/paper_r6/wavefield_kan_st_overthrust_r5.png
figures/paper_r6/wavefield_kan_st_overthrust_r6.png
figures/paper_r6/wavefield_kan_threelayer_r5.png
figures/paper_r6/wavefield_kan_threelayer_r6.png
figures/paper_r6/wavefield_kan_marmousi_r5.png
figures/paper_r6/wavefield_kan_marmousi_r6.png
figures/paper_r6/wavefield_kan_overthrust_r5.png
figures/paper_r6/wavefield_kan_overthrust_r6.png
```

当前源码中没有 `figures/paper_r6`、`wavefield_kan_`、`r5_vs_r6_t2_` 的生成代码；`rg` 只能在说明文档中搜到 `paper_r6`。从文件名可以推断：

- `wavefield_kan_st_<dataset>_r5.png`：KAN-ST-PINN 的 R5 波场图；
- `wavefield_kan_st_<dataset>_r6.png`：KAN-ST-PINN 的 R6 波场图；
- `wavefield_kan_<dataset>_r5.png`：Pure-KAN / KAN-PINN 的 R5 波场图；
- `wavefield_kan_<dataset>_r6.png`：Pure-KAN / KAN-PINN 的 R6 波场图；
- `r5_vs_r6_t2_<dataset>_kan_st.png`：KAN-ST-PINN 在 t=2 附近的 R5/R6 对比图。

但这些只是基于文件名和图像尺寸的推断。结论：`figures/paper_r6/` 不是当前 `main.py` 流水线的直接输出，更像是 R6 完成后的额外论文图脚本或未纳入仓库的后处理脚本生成。

## 7. 总结

能明确追到 `main.py` 的图片有五类：

1. `figures/<dataset>/kan_st_pinn/`：阶段 1，`train_kan_pinn.py` 训练 KAN-ST-PINN 后生成。
2. `figures/<dataset>/kan_pinn/`：阶段 2，`train_kan_pinn.py --pure-kan` 训练 Pure-KAN 后生成。
3. `figures/<dataset>/<st_variant>/`：阶段 3，`train_st_pinn.py --ablation` 训练 ST-PINN 消融模型后生成；当前目录只保留/产出了 `threelayer/full` 和 `threelayer/no_fourier`。
4. `figures/<dataset>/*_r6/`：阶段 4，`finetune.py` R6 微调后生成。
5. `figures/paper/` 和 `figures/ablation/`：阶段 6，分别由 `vis_compare.py` 和 `plot_abl.py` 生成。

不能明确追到当前 `main.py` 的图片：

- `figures/threelayer/full/loss_analysis.png`：来自 `hist_analysis.py`。
- `figures/threelayer/full_T2s/`：来自 `predict_kan_st_t25.py` 或其早期/带参数运行版本。
- `figures/final/`：当前源码没有生成逻辑，只能判断为外部/手工整理/未纳入脚本的最终图。
- `figures/paper_r6/`：当前源码没有生成逻辑，只能判断为 R6 后处理论文图或未纳入脚本生成。
