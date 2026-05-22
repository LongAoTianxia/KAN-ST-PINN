import os, sys, torch, numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, '.')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HIST_PATH = 'H:/wusheng/16/trained/threelayer/full/history.pt'
CKPT_DIR = 'H:/wusheng/16/trained/threelayer/full'
OUT_DIR = 'H:/wusheng/16/figures/threelayer/full'

h = torch.load(HIST_PATH, weights_only=False)
n = len(h['loss_total'])
print(f'Total epochs recorded: {n}')

# 打印关键指标
for key in ['loss_total', 'loss_pde', 'loss_ini', 'loss_bc', 'loss_data']:
    vals = h[key]
    print(f'{key}: first={vals[0]:.3e}  last={vals[-1]:.3e}  min={min(vals):.3e} @epoch{np.argmin(vals)+1}')

# 每500轮打印一次
print('\n=== Loss at milestones ===')
header = f'{"Epoch":>6} {"Total":>12} {"PDE":>12} {"Ini":>12} {"BC":>12} {"Data":>12}'
print(header)
print('-' * len(header))
milestones = list(range(99, n, 500)) + [n-1]
milestones = sorted(set(milestones))
for idx in milestones:
    print(f'{idx+1:>6} {h["loss_total"][idx]:>12.3e} {h["loss_pde"][idx]:>12.3e} '
          f'{h["loss_ini"][idx]:>12.3e} {h["loss_bc"][idx]:>12.3e} {h["loss_data"][idx]:>12.3e}')

# 画loss曲线
os.makedirs(OUT_DIR, exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 总损失
ax = axes[0, 0]
ax.plot(h['loss_total'], alpha=0.3, color='blue')
# 平滑曲线
window = min(100, n//10)
if window > 1:
    smooth = np.convolve(h['loss_total'], np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, window-1+len(smooth)), smooth, color='blue', linewidth=2)
ax.set_yscale('log')
ax.set_title('Total Loss')
ax.set_xlabel('Epoch')
ax.grid(True, alpha=0.3)

# 各分量损失
ax = axes[0, 1]
for key, label in [('loss_pde', 'PDE'), ('loss_ini', 'Initial'), ('loss_bc', 'BC'), ('loss_data', 'Data')]:
    vals = h[key]
    if any(v > 0 for v in vals):
        ax.plot(vals, label=label, alpha=0.5)
ax.set_yscale('log')
ax.set_title('Individual Losses')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# PDE损失放大（最后3000轮）
ax = axes[1, 0]
start = max(0, n - 3000)
ax.plot(range(start, n), h['loss_pde'][start:], alpha=0.5, color='red')
if window > 1 and n - start > window:
    vals = h['loss_pde'][start:]
    smooth = np.convolve(vals, np.ones(window)/window, mode='valid')
    ax.plot(range(start+window-1, start+window-1+len(smooth)), smooth, color='red', linewidth=2)
ax.set_yscale('log')
ax.set_title(f'PDE Loss (last {n-start} epochs)')
ax.set_xlabel('Epoch')
ax.grid(True, alpha=0.3)

# 数据+初始条件损失
ax = axes[1, 1]
ax.plot(h['loss_data'], label='Data', alpha=0.5)
ax.plot(h['loss_ini'], label='Initial', alpha=0.5)
ax.set_yscale('log')
ax.set_title('Data & Initial Losses')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT_DIR, 'loss_analysis.png')
plt.savefig(path, dpi=150)
plt.close()
print(f'\nSaved: {path}')

# 评估最新checkpoint
print('\n=== Evaluating model ===')
from train_st_pinn import (Config, prepare_dataset, align_temporal_config,
                           create_model, evaluate_model, save_comparison_plot)

config = Config()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
data = prepare_dataset(config, device)
align_temporal_config(config, data)

model = create_model(config, device, variant='full')

# 加载最新checkpoint
ckpts = [f for f in os.listdir(CKPT_DIR) if f.startswith('checkpoint_epoch') and f.endswith('.pt')]
ckpts.sort(key=lambda x: int(x.replace('checkpoint_epoch', '').replace('.pt', '')))
if ckpts:
    last_ckpt = ckpts[-1]
    print(f'Loading: {last_ckpt}')
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, last_ckpt), weights_only=True))
else:
    print('Loading: model.pt')
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'model.pt'), weights_only=True))

metrics = evaluate_model(model, data, device)
print(f'Relative L2 error: {metrics["rel_l2"]:.6f}')
print(f'MAE: {metrics["mae"]:.6f}')

save_comparison_plot(metrics['p_est'], data, OUT_DIR)
print('Done!')
