#!/usr/bin/env python3
"""R6微调: 加载R5权重继续训练, 加强晚期权重+更多数据点+低学习率"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(__file__))

from train_kan_pinn import (
    Config, set_seed, prepare_dataset, align_temporal_config,
    prepare_initial_condition, create_model, train_model,
    evaluate_model, save_loss_curves, save_comparison_plot
)


class R6Config(Config):
    """R6微调配置"""
    n_epochs = 4000
    learning_rate = 2e-4
    warmup_epochs = 200
    lr_min_ratio = 0.05       # decay to 1e-5

    n_data = 8000              # up from 5000
    n_colloc = 6000            # up from 4000
    n_bc = 1200                # up from 800

    # PDE直接开启（不需要预热）
    pde_warmup_epochs = 0
    pde_rampup_epochs = 0

    # 加强晚期权重
    late_data_alpha = 10.0     # up from 3.0 → 11x at t=T
    late_data_power = 3        # cubic instead of quadratic (steeper ramp)
    late_colloc_uniform = 0.7  # up from 0.5 (more late-time PDE coverage)

    # 收紧因果eps (model can handle it since already trained)
    causal_eps = 1.0           # was 3.0
    causal_eps_final = 0.3     # was 0.5


def main():
    parser = argparse.ArgumentParser(description='R6 fine-tuning from R5 weights')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['threelayer', 'marmousi', 'overthrust'])
    parser.add_argument('--pure-kan', action='store_true')
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--n-data', type=int, default=None)
    parser.add_argument('--n-colloc', type=int, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume R6 fine-tuning from latest checkpoint in output dir')
    args = parser.parse_args()

    config = R6Config()
    config.DATASET = args.dataset
    config.pure_kan = args.pure_kan
    config.n_epochs = args.epochs
    if args.n_data is not None:
        config.n_data = args.n_data
    if args.n_colloc is not None:
        config.n_colloc = args.n_colloc

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 源目录和输出目录
    src_tag = "kan_pinn" if config.pure_kan else "kan_st_pinn"
    dst_tag = f"{src_tag}_r6"
    src_dir = os.path.join(config.MODEL_DIR, config.DATASET, src_tag)
    out_dir = os.path.join(config.MODEL_DIR, config.DATASET, dst_tag)
    fig_dir = os.path.join(config.FIGURES_DIR, config.DATASET, dst_tag)

    set_seed(0)
    data = prepare_dataset(config, device)
    if not config.pure_kan:
        align_temporal_config(config, data)

    r_ini, p_ini = prepare_initial_condition(data, config, device)
    model = create_model(config, device)
    start_epoch = 0
    optimizer_state = None
    history_prev = None

    # 优先恢复R6 checkpoint；若不存在则从R5权重启动
    resumed = False
    if args.resume and os.path.isdir(out_dir):
        ckpts_r6 = sorted(
            [f for f in os.listdir(out_dir) if f.startswith('checkpoint_epoch') and f.endswith('.pt')],
            key=lambda f: int(f.replace('checkpoint_epoch', '').replace('.pt', ''))
        )
        if ckpts_r6:
            latest = ckpts_r6[-1]
            ckpt_path = os.path.join(out_dir, latest)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer_state = ckpt.get('optimizer_state')
            history_prev = ckpt.get('history')
            start_epoch = int(ckpt.get('epoch', latest.replace('checkpoint_epoch', '').replace('.pt', '')))
            resumed = True
            print(f"Resumed R6 checkpoint from {ckpt_path} (epoch {start_epoch})")

    if not resumed:
        model_path = os.path.join(src_dir, 'model.pt')
        if not os.path.exists(model_path):
            # 找最新checkpoint
            import glob
            ckpts = sorted(glob.glob(os.path.join(src_dir, 'checkpoint_epoch*.pt')),
                           key=lambda f: int(os.path.basename(f).replace('checkpoint_epoch','').replace('.pt','')))
            if ckpts:
                model_path = ckpts[-1]
            else:
                raise FileNotFoundError(f"No R5 model found in {src_dir}")

        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state' in state:
            model.load_state_dict(state['model_state'])
        else:
            model.load_state_dict(state)
        print(f"Loaded R5 weights from {model_path}")

    # 用R6配置继续训练
    history = train_model(model, data, r_ini, p_ini, config, device,
                          save_dir=out_dir, start_epoch=start_epoch,
                          optimizer_state=optimizer_state, history_prev=history_prev)

    # 保存最终模型
    os.makedirs(out_dir, exist_ok=True)
    torch.save({'model_state': model.state_dict()}, os.path.join(out_dir, 'model.pt'))
    torch.save(history, os.path.join(out_dir, 'history.pt'))

    # 评估
    metrics = evaluate_model(model, data, device)
    model_name = 'KAN-PINN' if config.pure_kan else 'KAN-ST-PINN'
    print(f"\n[{model_name} R6] Rel L2 = {metrics['rel_l2']:.6f}  MAE = {metrics['mae']:.6f}")

    save_loss_curves(history, fig_dir)
    save_comparison_plot(metrics['p_est'], data, fig_dir)


if __name__ == '__main__':
    main()
