#!/bin/bash
#SBATCH --job-name=registre-lora
#SBATCH --account=def-REMPLACER      # votre allocation (def-<PI>)
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out

module load python/3.11 cuda arrow
source $HOME/venvs/lf/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # nœuds de calcul sans internet

# copie locale rapide des images (disque du nœud)
cp -r $SCRATCH/registre_finetune $SLURM_TMPDIR/
sed "s#\${SCRATCH}/registre_finetune#$SLURM_TMPDIR/registre_finetune#; s#\${SCRATCH}#$SCRATCH#g" \
    scripts/narval/train_lora.yaml > $SLURM_TMPDIR/train.yaml

llamafactory-cli train $SLURM_TMPDIR/train.yaml
