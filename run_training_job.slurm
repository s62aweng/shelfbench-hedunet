#!/bin/bash
#SBATCH --job-name=hed-unet-training_2
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
# cpus-per-task=32
# mem=128G
# time=48:00:00

# Projektverzeichnis
PROJECT_DIR="/dss/dsshome1/07/di97toj/Git/hed-unet/ai-core-uc3"

# Micromamba-Modul laden
module load cuda/12.6.2
module load micromamba

# Shell initialisieren
eval "$(micromamba shell hook --shell bash)"

# Umgebung aktivieren
micromamba activate shelfbench-hedunet-env

# Sicherheitstest: Python vorhanden?
if ! command -v python &> /dev/null; then
    echo "❌ Python nicht verfügbar. Umgebung nicht korrekt aktiviert."
    exit 1
fi

# CUDA-Geräte anzeigen
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Daten vorbereiten
python "$PROJECT_DIR/prepare_data.py"

# Training starten
#python "$PROJECT_DIR/test.py" --config "$PROJECT_DIR/config.yml"

