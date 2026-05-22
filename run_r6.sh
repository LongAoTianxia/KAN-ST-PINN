#!/bin/bash
# Round 6 Launch: Fine-tune R5 models with enhanced late-time focus
# Run on server: bash /root/round6.sh
set -e

cd /root/PINNs_util

echo "=========================================="
echo " Round 6: Fine-tuning from R5 weights"
echo "=========================================="

# Clean old R6 checkpoints
for ds in threelayer marmousi overthrust; do
    for tag in kan_st_pinn_r6 kan_pinn_r6; do
        rm -rf ./trained/${ds}/${tag}
        mkdir -p ./trained/${ds}/${tag}
    done
done
echo "Cleaned R6 output directories."

# Launch 6 experiments on 6 GPUs
# GPU 0-2: KAN-ST-PINN (threelayer, marmousi, overthrust)
# GPU 3-5: Pure-KAN (threelayer, marmousi, overthrust)

# KAN-ST-PINN: use n_data=5000, n_colloc=4500 to fit in 24GB VRAM
CUDA_VISIBLE_DEVICES=0 nohup python finetune.py --dataset threelayer --epochs 4000 --n-data 5000 --n-colloc 4500 \
    > /root/r6_kan_st_threelayer.log 2>&1 &
echo "Started KAN-ST/threelayer on GPU 0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 nohup python finetune.py --dataset marmousi --epochs 4000 --n-data 5000 --n-colloc 4500 \
    > /root/r6_kan_st_marmousi.log 2>&1 &
echo "Started KAN-ST/marmousi on GPU 1 (PID $!)"

CUDA_VISIBLE_DEVICES=2 nohup python finetune.py --dataset overthrust --epochs 4000 --n-data 5000 --n-colloc 4500 \
    > /root/r6_kan_st_overthrust.log 2>&1 &
echo "Started KAN-ST/overthrust on GPU 2 (PID $!)"

# Pure-KAN: smaller model, can use full n_data=8000, n_colloc=6000
CUDA_VISIBLE_DEVICES=3 nohup python finetune.py --dataset threelayer --pure-kan --epochs 4000 \
    > /root/r6_kan_threelayer.log 2>&1 &
echo "Started Pure-KAN/threelayer on GPU 3 (PID $!)"

CUDA_VISIBLE_DEVICES=4 nohup python finetune.py --dataset marmousi --pure-kan --epochs 4000 \
    > /root/r6_kan_marmousi.log 2>&1 &
echo "Started Pure-KAN/marmousi on GPU 4 (PID $!)"

CUDA_VISIBLE_DEVICES=5 nohup python finetune.py --dataset overthrust --pure-kan --epochs 4000 \
    > /root/r6_kan_overthrust.log 2>&1 &
echo "Started Pure-KAN/overthrust on GPU 5 (PID $!)"

echo ""
echo "All 6 R6 experiments launched. Monitor with: python /root/check_r6.py"
