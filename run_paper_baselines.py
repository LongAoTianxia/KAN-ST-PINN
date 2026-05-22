import argparse
import os
import subprocess
import sys
import time


DATASETS = ("threelayer", "marmousi", "overthrust")

JOBS = [
    {
        "name": "FF-PINN",
        "script": "train_ff_pinn.py",
        "tag": "ff_pinn_paper",
        "args": [],
    },
    {
        "name": "Gabor-Enhanced-PINN",
        "script": "train_gepinn.py",
        "tag": "gabor_enhanced_pinn",
        "args": [],
    },
    {
        "name": "PINNsFormer",
        "script": "train_pinnsformer.py",
        "tag": "pinnsformer_paper",
        "args": [],
    },
    {
        "name": "Raissi-PINN",
        "script": "train_raissi_pinn.py",
        "tag": "raissi_pinn",
        "args": [],
    },
    {
        "name": "TD-PINN",
        "script": "train_tdpinn.py",
        "tag": "td_pinn_paper",
        "args": [],
    },
]


def model_path(root, dataset, tag):
    return os.path.join(root, "trained", dataset, tag, "model.pt")


def missing_datasets(root, tag):
    return [dataset for dataset in DATASETS if not os.path.exists(model_path(root, dataset, tag))]


def run_command(cmd, cwd, dry_run=False):
    print("\n" + "=" * 100, flush=True)
    print("RUN:", " ".join(cmd), flush=True)
    print("=" * 100, flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def train_job(job, root, python_exe, dry_run=False):
    missing = missing_datasets(root, job["tag"])
    if not missing:
        print(f"[SKIP] {job['name']}: all model.pt files already exist.", flush=True)
        return

    print(f"[TRAIN] {job['name']}: missing {', '.join(missing)}", flush=True)
    for dataset in missing:
        cmd = [python_exe, job["script"], "--dataset", dataset] + job["args"]
        run_command(cmd, root, dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paper-setting baselines sequentially, skipping existing model.pt files."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for child scripts. With conda run, default is the d2l Python.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final compute_l2.py evaluation.",
    )
    parser.add_argument(
        "--eval-skip-pde",
        action="store_true",
        help="Pass --skip-pde to compute_l2.py for quick final evaluation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    started = time.time()

    print(f"Project root: {root}", flush=True)
    print(f"Child python: {args.python}", flush=True)

    for job in JOBS:
        train_job(job, root, args.python, dry_run=args.dry_run)

    if not args.skip_eval:
        eval_cmd = [args.python, "compute_l2.py"]
        if args.eval_skip_pde:
            eval_cmd.append("--skip-pde")
        run_command(eval_cmd, root, dry_run=args.dry_run)

    elapsed = time.time() - started
    print(f"\nDone. Total orchestration time: {elapsed:.1f}s ({elapsed / 3600:.3f}h)", flush=True)


if __name__ == "__main__":
    main()
