# KAN-ST-PINN for 2D Acoustic Wave Modeling

This repository contains PyTorch implementations for solving 2D acoustic wave equations with physics-informed neural networks. The main model is **KAN-ST-PINN**, which combines a KAN spatial encoder with temporal sequence modeling and attention for wavefield prediction.

The project includes training, fine-tuning, evaluation, and visualization scripts for synthetic layered and complex velocity models.

## Features

- KAN-ST-PINN for spatiotemporal wavefield modeling
- Pure KAN-PINN baseline
- ST-PINN ablation experiments
- FF-PINN, Gabor-PINN, PINNsFormer, TD-PINN, and Raissi-style baseline scripts
- Finite-difference reference wavefield generation
- Relative L2 / MAE evaluation and paper-style visualization utilities

## Repository Layout

```text
.
├── main.py                     # End-to-end experiment pipeline
├── train_kan_pinn.py           # KAN-ST-PINN and Pure KAN-PINN training
├── finetune.py                 # Fine-tuning from trained checkpoints
├── compute_l2.py               # Evaluation metrics
├── evaluate.py                 # Fine-tuning evaluation helper
├── vis_compare.py              # Paper-style visualizations
├── plot_abl.py                 # KAN-ST vs Pure-KAN comparison plots
├── train_*.py                  # Additional baseline training scripts
└── PINNs_util/
    ├── datasets.py             # Dataset/reference data generation
    ├── PINNs_fdiff.py          # Finite-difference solver
    ├── PINNs_aux.py            # PDE residuals, sampling, and utilities
    ├── kan_pinn.py             # KAN-PINN and KAN-ST-PINN models
    └── st_pinn.py              # ST-PINN model
```

Generated data, figures, checkpoints, logs, and paper PDFs are intentionally ignored by Git.

## Datasets

The default experiments use three synthetic 2D velocity models:

- `threelayer`
- `marmousi`
- `overthrust`

Generate reference data with:

```bash
python PINNs_util/datasets.py
```

Some velocity-surrogate experiments require extracted velocity crops:

```bash
python PINNs_util/datasets.py --extracted
```

## Quick Start

Install the usual scientific Python stack with PyTorch, NumPy, Matplotlib, and tqdm. A CUDA-capable GPU is recommended for full training.

Run a short pipeline check:

```bash
python main.py --quick
```

Train KAN-ST-PINN on one dataset:

```bash
python train_kan_pinn.py --dataset threelayer
```

Train the Pure KAN baseline:

```bash
python train_kan_pinn.py --dataset threelayer --pure-kan
```

Evaluate trained models:

```bash
python compute_l2.py
```

Generate summary figures:

```bash
python vis_compare.py
python plot_abl.py
```

## Main Pipeline

`main.py` orchestrates the standard workflow:

1. Train KAN-ST-PINN
2. Train Pure KAN-PINN
3. Run ST-PINN ablations
4. Fine-tune trained KAN models
5. Evaluate metrics
6. Generate visualizations

Useful modes:

```bash
python main.py --eval-only
python main.py --finetune-only
python main.py --steps train_kan_st train_kan evaluate visualize
python main.py --resume-step 4
```

## Outputs

Common output locations:

```text
datasets/   # generated .npz reference datasets
trained/    # model checkpoints and histories
figures/    # loss curves, wavefield comparisons, and summary plots
```

These directories are not tracked by Git. Re-run the data generation and training scripts to reproduce them.

## Notes

- Full experiments are computationally expensive.
- Model architecture constants in evaluation scripts should match the training configuration used for a checkpoint.
- The repository contains historical and experimental scripts; `train_kan_pinn.py`, `compute_l2.py`, and `vis_compare.py` are the most relevant entry points for the current KAN-ST-PINN workflow.
