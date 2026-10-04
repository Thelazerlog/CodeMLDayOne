#!/bin/bash
# Lance vLLM sur le nœud GPU (modèle de base, ou de base + adaptateur LoRA), puis exécute la commande
# passée en argument contre ce serveur local. Données SYNTHÉTIQUES uniquement.
#
#   sbatch --account=def-XXX_gpu scripts/narval/vlm_job.sbatch python -m registre.cli evaluate data/augmented --out out/base
#   LORA=$SCRATCH/runs/registre_lora_nuit sbatch --account=def-XXX_gpu scripts/narval/vlm_job.sbatch \
#        python -m registre.cli evaluate data/augmented --glob "patient_[01][09]_*" --out out/lora_test
#   sbatch --account=def-XXX_gpu scripts/narval/vlm_job.sbatch python -m registre.cli record data/demo
#
#SBATCH --job-name=registre-vlm
#SBATCH --account=def-REMPLACER
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
MODEL=${MODEL:-Qwen3-VL-8B-Instruct}
GPUS=${GPUS:-1}
LORA=${LORA:-}

module load python/3.11 cuda arrow opencv
source $HOME/venvs/vllm/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# avec LoRA : le modèle de base s'appelle « base », l'adaptateur « registre » (celui qu'interroge le pipeline)
SERVED=registre
EXTRA=()
if [ -n "$LORA" ]; then
    SERVED=base
    EXTRA=(--enable-lora --max-lora-rank 16 --lora-modules "registre=$LORA")
fi
vllm serve $SCRATCH/models/$MODEL --served-model-name $SERVED --port 8000 \
     --tensor-parallel-size $GPUS --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' \
     --gpu-memory-utilization 0.90 "${EXTRA[@]}" > vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
    curl -sf localhost:8000/v1/models >/dev/null && break
    kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM s'est arrêté : voir vllm-$SLURM_JOB_ID.log"; tail -40 vllm-$SLURM_JOB_ID.log; exit 1; }
    sleep 5
done
curl -s localhost:8000/v1/models

cat > $SLURM_TMPDIR/config.toml <<EOF
[vlm]
base_url = "http://localhost:8000/v1"
model = "registre"
backend = "openai"
batch = 8
parallel = 16
timeout = 300
EOF
export REGISTRE_CONFIG=$SLURM_TMPDIR/config.toml
echo "commande : $*"
"$@"
kill $VLLM_PID || true
