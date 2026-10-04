#!/bin/bash
# Prépare le jeu d'entraînement LoRA sur un nœud de CALCUL (CPU seulement, pas de GPU) :
# pages synthétiques (autres polices manuscrites, papier coloré...) -> photos dégradées -> mosaïques + JSON.
# Découpage sans fuite : patientes 1-7 entraînement, 8 validation, 9-10 test ; synthétiques -> entraînement.
#
#   sbatch --account=def-XXX_cpu scripts/narval/prep_data.sbatch
#
#SBATCH --job-name=registre-prep
#SBATCH --account=def-REMPLACER
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
N=${N:-1000}
module load python/3.11 arrow opencv
source $HOME/venvs/vllm/bin/activate

python -m registre.cli synth --n $N --out data/synth
python -m registre.cli augment --src data/synth --out data/augmented_synth --per-page 2
python -m registre.cli augment --src data/clean --out data/augmented_ft --per-page 6 --seed 1
python -m registre.cli finetune-dataset --src data/augmented_ft data/augmented_synth --out $SCRATCH/registre_finetune
du -sh $SCRATCH/registre_finetune
wc -l $SCRATCH/registre_finetune/*.jsonl
