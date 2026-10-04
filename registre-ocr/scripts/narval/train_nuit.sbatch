#!/bin/bash
# Entraînement LoRA court (voir train_lora_nuit.yaml). À lancer après prep_data.sbatch.
#
#   sbatch --account=def-XXX_gpu scripts/narval/train_nuit.sbatch
#
#SBATCH --job-name=registre-lora
#SBATCH --account=def-REMPLACER
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load python/3.11 cuda arrow
source $HOME/venvs/lf/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # nœuds de calcul sans internet
# les roues de l'Alliance portent un suffixe (« 5.6.0+computecanada ») que le contrôle de version de
# LLaMA-Factory refuse alors que la version est la bonne
export DISABLE_VERSION_CHECK=1

cp -r $SCRATCH/registre_finetune $SLURM_TMPDIR/
sed "s#\${SCRATCH}/registre_finetune#$SLURM_TMPDIR/registre_finetune#; s#\${SCRATCH}#$SCRATCH#g" \
    scripts/narval/train_lora_nuit.yaml > $SLURM_TMPDIR/train.yaml
cat $SLURM_TMPDIR/train.yaml
llamafactory-cli train $SLURM_TMPDIR/train.yaml
ls -la $SCRATCH/runs/registre_lora_nuit
