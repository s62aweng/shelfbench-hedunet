#!/bin/bash
#SBATCH --job-name=hedunet_shelfbench_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --exclude=hpdar04c03s03,hpdar04c02s04

# Projektverzeichnis
PROJECT_DIR="/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/git-project/shelf-bench-hedunet/shelfbench-hedunet"

# Module laden
module load cuda/12.6.2
module load micromamba

# Shell initialisieren
eval "$(micromamba shell hook --shell bash)"

# Umgebung aktivieren
#micromamba activate shelfbench-hedunet-env
micromamba activate shelfbench-hedunet-py311


# Sicherheitstest: Python vorhanden?
if ! command -v python &> /dev/null; then
    echo "❌ Python nicht verfügbar. Umgebung nicht korrekt aktiviert."
    exit 1
fi

# CUDA-Geräte anzeigen
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Training starten
python "$PROJECT_DIR/ideal_train_file.py" \
    --config-path "$PROJECT_DIR/conf" \
    --config-name config \
    model_name=HEDUNet \
    loss_function=HEDUNetLoss

#python "$PROJECT_DIR/ideal_train_file.py" \
#    --config "$PROJECT_DIR/conf/config.yml" \
#    --model_name "HEDUNet" \
#    --loss_function "HEDUNetLoss"



