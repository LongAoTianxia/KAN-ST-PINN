import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import argparse
import subprocess
import time

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 当前仓库中的训练/评估/出图脚本都位于项目根目录，
# 不能把 cwd 设到 PINNs_util，否则子进程会找不到脚本。
SCRIPT_DIR = ROOT_DIR

DATASETS = ["threelayer", "marmousi", "overthrust"]
ST_PINN_VARIANTS = ["full", "no_fourier", "no_lstm", "no_attention", "no_lstm_attn", "self_attn"]


def _checkpoint_exists(model_dir):
    if not os.path.isdir(model_dir):
        return False
    return any(
        name.startswith("checkpoint_epoch") and name.endswith(".pt")
        for name in os.listdir(model_dir)
    )


def _model_or_checkpoint_exists(model_dir):
    return os.path.exists(os.path.join(model_dir, "model.pt")) or _checkpoint_exists(model_dir)


def _missing_for_step(step_key):
    """返回某一步已完成所需产物的缺失列表。空列表表示视为已完成。"""
    missing = []

    if step_key == "train_kan_st":
        for ds in DATASETS:
            model_dir = os.path.join(ROOT_DIR, "trained", ds, "kan_st_pinn")
            if not _model_or_checkpoint_exists(model_dir):
                missing.append(f"缺少 KAN-ST-PINN 产物: {model_dir}")

    elif step_key == "train_kan":
        for ds in DATASETS:
            model_dir = os.path.join(ROOT_DIR, "trained", ds, "kan_pinn")
            if not _model_or_checkpoint_exists(model_dir):
                missing.append(f"缺少 Pure-KAN 产物: {model_dir}")

    elif step_key == "train_st":
        for ds in DATASETS:
            for variant in ST_PINN_VARIANTS:
                out_dir = os.path.join(ROOT_DIR, "trained", ds, variant)
                model_path = os.path.join(out_dir, "model.pt")
                hist_path = os.path.join(out_dir, "history.pt")
                if not os.path.exists(model_path):
                    missing.append(f"缺少 ST-PINN 模型: {model_path}")
                if not os.path.exists(hist_path):
                    missing.append(f"缺少 ST-PINN 历史: {hist_path}")
            summary_path = os.path.join(ROOT_DIR, "figures", ds, "ablation_summary.txt")
            if not os.path.exists(summary_path):
                missing.append(f"缺少 ST-PINN 消融汇总: {summary_path}")

    elif step_key == "finetune":
        for ds in DATASETS:
            for tag in ("kan_st_pinn", "kan_pinn"):
                model_dir = os.path.join(ROOT_DIR, "trained", ds, tag)
                if not _model_or_checkpoint_exists(model_dir):
                    missing.append(f"缺少微调前基础模型: {model_dir}")
            for tag in ("kan_st_pinn_r6", "kan_pinn_r6"):
                model_path = os.path.join(ROOT_DIR, "trained", ds, tag, "model.pt")
                if not os.path.exists(model_path):
                    missing.append(f"缺少 R6 模型: {model_path}")

    elif step_key == "evaluate":
        for ds in DATASETS:
            for tag in ("kan_st_pinn", "kan_pinn"):
                model_dir = os.path.join(ROOT_DIR, "trained", ds, tag)
                if not _model_or_checkpoint_exists(model_dir):
                    missing.append(f"缺少主线模型用于评测: {model_dir}")
            for tag in ("kan_st_pinn_r6", "kan_pinn_r6"):
                model_path = os.path.join(ROOT_DIR, "trained", ds, tag, "model.pt")
                if not os.path.exists(model_path):
                    missing.append(f"缺少 R6 模型用于评测: {model_path}")
        result_json = os.path.join(ROOT_DIR, "r6_results.json")
        if not os.path.exists(result_json):
            missing.append(f"缺少评测汇总结果: {result_json}")

    elif step_key == "visualize":
        for ds in DATASETS:
            for tag in ("kan_st_pinn", "kan_pinn"):
                model_dir = os.path.join(ROOT_DIR, "trained", ds, tag)
                if not _model_or_checkpoint_exists(model_dir):
                    missing.append(f"缺少可视化所需模型: {model_dir}")
                hist_path = os.path.join(model_dir, "history.pt")
                if not os.path.exists(hist_path):
                    missing.append(f"缺少可视化所需历史: {hist_path}")
        full_hist = os.path.join(ROOT_DIR, "trained", "threelayer", "full", "history.pt")
        full_hist_6000 = os.path.join(ROOT_DIR, "trained", "threelayer", "full", "history_6000.pt")
        if not (os.path.exists(full_hist) or os.path.exists(full_hist_6000)):
            missing.append("缺少 vis_compare.py 所需的 threelayer/full 历史文件")

    return missing


def _stage_number_to_key(stage_no):
    if stage_no < 1 or stage_no > len(DEFAULT_PIPELINE):
        raise ValueError(f"阶段序号必须在 1 到 {len(DEFAULT_PIPELINE)} 之间")
    return DEFAULT_PIPELINE[stage_no - 1]


def _can_resume_from(stage_no):
    stage_key = _stage_number_to_key(stage_no)
    # 续跑依赖按实验真实输入约束，而不是机械按流水线顺序要求全部前置步骤。
    resume_deps = {
        1: [],
        2: [],
        3: [],
        4: ["train_kan_st", "train_kan"],
        5: ["train_kan_st", "train_kan", "finetune"],
        6: ["train_kan_st", "train_kan"],
    }
    deps = resume_deps[stage_no]
    missing = []
    for dep in deps:
        dep_missing = _missing_for_step(dep)
        if dep_missing:
            missing.append(f"[前置阶段未完成] {dep}")
            missing.extend(dep_missing)
    return stage_key, missing


def _resume_requested_for(args, step_key):
    return getattr(args, "resume_step", None) is not None and _stage_number_to_key(args.resume_step) == step_key


def run_cmd(cmd, cwd=None, desc=""):
    """运行子进程，实时打印输出"""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=cwd or ROOT_DIR,
        stdout=sys.stdout, stderr=sys.stderr, text=True,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[错误] {desc} 返回码={proc.returncode}  (耗时 {elapsed:.1f}s)")
        return False
    print(f"\n[完成] {desc}  (耗时 {elapsed:.1f}s)")
    return True


def step_train_kan_st(args):
    for ds in DATASETS:
        cmd = [
            sys.executable, "train_kan_pinn.py",
            "--dataset", ds,
            "--epochs", str(args.epochs),
        ]
        if args.ablation:
            cmd.append("--ablation")
        if _resume_requested_for(args, "train_kan_st"):
            cmd.append("--resume")
        desc = f"训练 KAN-ST-PINN 消融 / {ds}" if args.ablation else f"训练 KAN-ST-PINN / {ds}"
        if not run_cmd(cmd, cwd=SCRIPT_DIR, desc=desc):
            return False
    return True


def step_train_kan_pure(args):
    for ds in DATASETS:
        cmd = [
            sys.executable, "train_kan_pinn.py",
            "--dataset", ds, "--pure-kan",
            "--epochs", str(args.epochs),
        ]
        if _resume_requested_for(args, "train_kan"):
            cmd.append("--resume")
        if not run_cmd(cmd, cwd=SCRIPT_DIR, desc=f"训练 Pure-KAN / {ds}"):
            return False
    return True


def step_train_st_pinn(args):
    for ds in DATASETS:
        cmd = [
            sys.executable, "train_st_pinn.py",
            "--dataset", ds, "--ablation",
            "--epochs", str(args.epochs),
        ]
        if not run_cmd(cmd, cwd=SCRIPT_DIR, desc=f"训练 ST-PINN 消融 / {ds}"):
            return False
    return True


def step_finetune(args):
    for ds in DATASETS:
        for extra in [[], ["--pure-kan"]]:
            tag = "Pure-KAN" if extra else "KAN-ST-PINN"
            n_data = 8000 if extra else 5000
            n_colloc = 6000 if extra else 4500
            cmd = [
                sys.executable, "finetune.py",
                "--dataset", ds,
                "--epochs", str(args.finetune_epochs),
                "--n-data", str(n_data),
                "--n-colloc", str(n_colloc),
            ] + extra
            if _resume_requested_for(args, "finetune"):
                cmd.append("--resume")
            if not run_cmd(cmd, cwd=SCRIPT_DIR, desc=f"微调 {tag} / {ds}"):
                return False
    return True


def step_evaluate():
    ok1 = run_cmd(
        [sys.executable, "compute_l2.py"],
        cwd=ROOT_DIR, desc="评测: 计算 R5 / 主线模型 Rel L2"
    )
    ok2 = run_cmd(
        [sys.executable, "evaluate.py"],
        cwd=ROOT_DIR, desc="评测: 检查 R6 微调结果"
    )
    return ok1 and ok2


def step_visualize():
    """步骤3: 生成论文图表"""
    ok1 = run_cmd(
        [sys.executable, "vis_compare.py"],
        cwd=ROOT_DIR, desc="可视化: 论文图表"
    )
    ok2 = run_cmd(
        [sys.executable, "plot_abl.py"],
        cwd=ROOT_DIR, desc="可视化: 消融对比图"
    )
    return ok1 and ok2



STEPS = {
    "train_kan_st":  ("训练 KAN-ST-PINN",     step_train_kan_st),
    "train_kan":     ("训练 Pure-KAN",         step_train_kan_pure),
    "train_st":      ("训练 ST-PINN (消融)",   step_train_st_pinn),
    "finetune":      ("微调 (R6)",             step_finetune),
    "evaluate":      ("评测 Rel L2",           step_evaluate),
    "visualize":     ("生成图表",              step_visualize),
}

# 默认执行顺序: 训练 → 微调 → 评测 → 出图
DEFAULT_PIPELINE = ["train_kan_st", "train_kan", "train_st", "finetune", "evaluate", "visualize"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="KAN-ST-PINN 完整实验流程",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=list(STEPS.keys()),
        default=None,
        help="指定要执行的步骤 (默认: 全部训练+评测+出图)\n"
             "可选: " + ", ".join(STEPS.keys()),
    )
    parser.add_argument("--epochs", type=int, default=10000,
                        help="训练轮数 (默认: 10000)")
    parser.add_argument("--finetune-epochs", type=int, default=4000,
                        help="微调轮数 (默认: 4000)")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (epochs=600)")
    parser.add_argument("--ablation", action="store_true",
                        help="在 train_kan_st 阶段运行 KAN-ST temporal ablations")
    parser.add_argument("--eval-only", action="store_true",
                        help="只跑评测+出图 (跳过训练)")
    parser.add_argument("--finetune-only", action="store_true",
                        help="只跑微调+评测+出图")
    parser.add_argument("--resume-step", type=int, default=None,
                        help="按默认流水线阶段序号续跑。"
                             f"可选 1-{len(DEFAULT_PIPELINE)}，"
                             "例如 4 表示从 finetune 开始。")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.epochs = 600
        args.finetune_epochs = 200

    if args.resume_step is not None and (args.steps or args.eval_only or args.finetune_only):
        raise SystemExit("--resume-step 不能与 --steps / --eval-only / --finetune-only 同时使用")

    # 确定要执行的步骤
    if args.resume_step is not None:
        stage_key, missing = _can_resume_from(args.resume_step)
        if missing:
            print("=" * 60)
            print(f"  无法从阶段 {args.resume_step} ({stage_key}) 续跑")
            print("=" * 60)
            for item in missing:
                print(f"  - {item}")
            raise SystemExit(1)
        pipeline = DEFAULT_PIPELINE[args.resume_step - 1:]
    elif args.steps:
        pipeline = args.steps
    elif args.eval_only:
        pipeline = ["evaluate", "visualize"]
    elif args.finetune_only:
        pipeline = ["finetune", "evaluate", "visualize"]
    else:
        pipeline = DEFAULT_PIPELINE

    print("=" * 60)
    print("  KAN-ST-PINN 实验流程")
    print("=" * 60)
    print(f"  训练轮数: {args.epochs}")
    print(f"  微调轮数: {args.finetune_epochs}")
    print(f"  KAN-ST 消融: {'开启' if args.ablation else '关闭'}")
    print("  阶段编号:")
    for idx, step in enumerate(DEFAULT_PIPELINE, 1):
        print(f"    {idx}. {step}")
    if args.resume_step is not None:
        print(f"  续跑起点: 第 {args.resume_step} 阶段 ({_stage_number_to_key(args.resume_step)})")
    print(f"  执行步骤:")
    for i, step in enumerate(pipeline, 1):
        name = STEPS[step][0]
        print(f"    {i}. {name}")
    print("=" * 60)

    t_total = time.time()
    failed = []

    for step_key in pipeline:
        step_name, step_func = STEPS[step_key]

        # evaluate / visualize 不需要 args 参数
        if step_key in ("evaluate", "visualize"):
            ok = step_func()
        else:
            ok = step_func(args)

        if not ok:
            failed.append(step_name)
            print(f"\n[警告] {step_name} 失败，继续下一步...")

    elapsed_total = time.time() - t_total

    # 汇总
    print(f"\n{'='*60}")
    print(f"  流程结束  (总耗时 {elapsed_total/3600:.2f} 小时)")
    print(f"{'='*60}")
    if failed:
        print(f"  失败步骤: {', '.join(failed)}")
    else:
        print("  全部成功!")
    print()


if __name__ == "__main__":
    main()
