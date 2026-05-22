#!/usr/bin/env python3
"""R6评估: 计算Rel L2并与R5对比"""
import os, sys, json, time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PINNs_util'))
# 复用已有的训练框架
from train_kan_pinn import (
    Config, set_seed, prepare_dataset, align_temporal_config,
    prepare_initial_condition, create_model, evaluate_model
)
from finetune import R6Config

R5_BASELINE = {
    'kan_st_pinn/threelayer':  0.027574,
    'kan_st_pinn/marmousi':    0.041026,
    'kan_st_pinn/overthrust':  0.037547,
    'kan_pinn/threelayer':     0.098174,
    'kan_pinn/marmousi':       0.118788,
    'kan_pinn/overthrust':     0.124966,
}

DATASETS = ['threelayer', 'marmousi', 'overthrust']


def eval_one(dataset, pure_kan, device):
    """评估单个R6模型"""
    config = R6Config()
    config.DATASET = dataset
    config.pure_kan = pure_kan

    src_tag = "kan_pinn" if pure_kan else "kan_st_pinn"
    r6_tag = f"{src_tag}_r6"
    model_path = os.path.join(config.MODEL_DIR, dataset, r6_tag, 'model.pt')

    if not os.path.exists(model_path):
        return None

    set_seed(0)
    data = prepare_dataset(config, device)
    if not pure_kan:
        align_temporal_config(config, data)
    model = create_model(config, device)

    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)

    metrics = evaluate_model(model, data, device)
    return metrics['rel_l2']


def wait_for_completion(max_wait=7200, poll_interval=60):
    """等所有模型训练完"""
    needed = []
    for ds in DATASETS:
        for tag in ['kan_st_pinn_r6', 'kan_pinn_r6']:
            p = os.path.join('./trained', ds, tag, 'model.pt')
            needed.append(p)

    start = time.time()
    while time.time() - start < max_wait:
        missing = [p for p in needed if not os.path.exists(p)]
        if not missing:
            print("All 6 R6 models found!")
            return True
        # 打印进度
        done = len(needed) - len(missing)
        elapsed = int(time.time() - start)
        # 读每个实验的状态
        for ds in DATASETS:
            for tag in ['kan_st_pinn_r6', 'kan_pinn_r6']:
                sf = os.path.join('./trained', ds, tag, 'status.txt')
                if os.path.exists(sf):
                    with open(sf) as f:
                        status = f.read().strip()
                else:
                    status = "N/A"
                mp = os.path.join('./trained', ds, tag, 'model.pt')
                mark = "DONE" if os.path.exists(mp) else "..."
                short = 'KAN-ST' if 'st' in tag else 'Pure-K'
                print(f"  [{elapsed:4d}s] {short}/{ds[:6]:6s} {status}  {mark}")
        print(f"  Waiting... {done}/{len(needed)} complete, checking again in {poll_interval}s\n")
        time.sleep(poll_interval)

    print("TIMEOUT: Not all models completed within max_wait")
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true', help='Wait for training completion first')
    parser.add_argument('--max-wait', type=int, default=7200)
    parser.add_argument('--poll', type=int, default=60)
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PINNs_util'))

    if args.wait:
        ok = wait_for_completion(args.max_wait, args.poll)
        if not ok:
            sys.exit(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 72)
    print("R6 EVALUATION RESULTS")
    print("=" * 72)

    results = {}
    all_improved = True
    any_worse = False

    for pure_kan in [False, True]:
        model_type = 'Pure-KAN' if pure_kan else 'KAN-ST'
        src_tag = 'kan_pinn' if pure_kan else 'kan_st_pinn'
        for ds in DATASETS:
            key = f"{src_tag}/{ds}"
            r5_val = R5_BASELINE.get(key, 999)

            rel_l2 = eval_one(ds, pure_kan, device)
            if rel_l2 is None:
                print(f"  {model_type:8s}/{ds:12s}  R6: NOT FOUND")
                all_improved = False
                continue

            change = (rel_l2 - r5_val) / r5_val * 100
            if rel_l2 < r5_val * 0.95:
                verdict = "PASS"
            elif rel_l2 < r5_val * 1.05:
                verdict = "MARGINAL"
            else:
                verdict = "FAIL"
                any_worse = True

            if rel_l2 >= r5_val:
                all_improved = False

            results[key] = {'r6': rel_l2, 'r5': r5_val, 'change': change, 'verdict': verdict}
            print(f"  {model_type:8s}/{ds:12s}  R5={r5_val:.6f}  R6={rel_l2:.6f}  "
                  f"change={change:+.1f}%  [{verdict}]")

    # 总体结论
    print("\n" + "-" * 72)
    if all_improved:
        overall = "SUCCESS"
    elif any_worse:
        overall = "NEEDS_ITERATION"
    else:
        overall = "MARGINAL"

    print(f"OVERALL: {overall}")

    # 保存结果json
    result_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r6_results.json')
    with open(result_file, 'w') as f:
        json.dump({'results': results, 'overall': overall, 'timestamp': time.time()}, f, indent=2)
    print(f"Results saved to {result_file}")


if __name__ == '__main__':
    main()
