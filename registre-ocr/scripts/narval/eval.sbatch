#!/bin/bash
# Évaluation complète sur Narval : vLLM sert le modèle sur le nœud de calcul (localhost),
# le pipeline l'interroge comme il interroge Ollama sur le Mac. Données SYNTHÉTIQUES uniquement.
#
#   sbatch scripts/narval/eval.sbatch data/dev  out/dev           # Qwen3-VL-8B (comme sur le Mac)
#   MODEL=Qwen3-VL-32B-Instruct GPUS=2 sbatch --gpus-per-node=a100:2 scripts/narval/eval.sbatch data/test_A out/test_A_32b
#
#SBATCH --job-name=registre-eval
#SBATCH --account=def-REMPLACER
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
IMAGES=${1:?dossier d'images}
OUT=${2:?dossier de sortie}
MODEL=${MODEL:-Qwen3-VL-8B-Instruct}
GPUS=${GPUS:-1}

module load python/3.11 cuda arrow opencv
source $HOME/venvs/vllm/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1. serveur vLLM local au nœud
vllm serve $SCRATCH/models/$MODEL --served-model-name registre --port 8000 \
     --tensor-parallel-size $GPUS --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' \
     --gpu-memory-utilization 0.90 > vllm-$SLURM_JOB_ID.log 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 5; done

# 2. évaluation (backend « openai » = protocole HTTP, sur localhost : rien ne sort du nœud)
cat > $SLURM_TMPDIR/config.toml <<EOF
[vlm]
base_url = "http://localhost:8000/v1"
model = "registre"
backend = "openai"
batch = 8
parallel = 16
timeout = 300
EOF
REGISTRE_CONFIG=$SLURM_TMPDIR/config.toml python -m registre.cli evaluate "$IMAGES" --reader vlm --out "$OUT"
