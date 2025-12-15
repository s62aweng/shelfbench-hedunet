#!/bin/bash
#SBATCH --job-name=hedunet_shelfbench_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=hpdar04c03s03,hpdar04c02s04

# Anzeige des Home-Verzeichnisses und .netrc-Inhalts zur Fehlerbehebung
echo $HOME
cat ~/.netrc

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

# Bibliothekspfad konfigurieren

# Bibliothekspfad strikt setzen (nur Conda-Forge Libs)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# Debug: welche libtiff/libjpeg werden von rasterio geladen?
echo "Check welche Bibliotheken rasterio/_base.so nutzt:"
ldd $CONDA_PREFIX/lib/python3.11/site-packages/rasterio/_base*.so | egrep "libtiff|jpeg"

# Debug: Rasterio-Version prüfen
python -c "import rasterio; print('Rasterio version:', rasterio.__version__)"

# Training starten
python "$PROJECT_DIR/ideal_train_file.py" \
    --config-path "$PROJECT_DIR/conf" \
    --config-name config \
    model.name=HEDUNet \
    training.loss_function=HEDUNetLoss



