#!/bin/bash
#SBATCH --job-name=hed-unet-inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Projektverzeichnis
PROJECT_DIR="/dss/dsshome1/07/di97toj/Git/hed-unet/ai-core-uc3"

# Micromamba-Modul laden
module load micromamba

# Shell initialisieren
eval "$(micromamba shell hook --shell bash)"

# Umgebung aktivieren
micromamba activate hed-unet-env

# Sicherheitstest: Python vorhanden?
if ! command -v python &> /dev/null; then
    echo "❌ Python nicht verfügbar. Umgebung nicht korrekt aktiviert."
    exit 1
fi

# Inferenz starten
# python "$PROJECT_DIR/inference.py" "$PROJECT_DIR/tiles/" --model "$PROJECT_DIR/default_model" -c 99
python "$PROJECT_DIR/inference.py" "/dss/dsstbyfs02/pn49ci/pn49ci-dss-0000/Antartic_Database/data/Inference_HED-UNet/tiles" --model "$PROJECT_DIR/default_model" -c 99