# Fine-tuning sur Narval (Calcul Québec)

Point de départ à vérifier contre la documentation de l'Alliance et la version de LLaMA-Factory
installée : les noms de modules, de comptes et d'options évoluent.

## Règle de confidentialité

Seules les données **synthétiques** du défi vont sur Narval. Si un jour de vraies photos de registres
existent, elles ne quittent pas l'infrastructure du système de santé concerné : ni Narval, ni aucun
service tiers.

## 1. Sur le nœud de connexion (accès internet)

Les nœuds de calcul n'ont pas d'accès internet. On télécharge donc tout avant de lancer le job.

```bash
module load python/3.11 cuda arrow
virtualenv --no-download $HOME/venvs/lf && source $HOME/venvs/lf/bin/activate
pip install --no-index --upgrade pip
pip install llamafactory            # si absent du wheelhouse de l'Alliance : pip install depuis PyPI
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir $SCRATCH/models/Qwen3-VL-8B-Instruct
```

Copier ensuite le jeu de données produit sur le Mac :

```bash
# sur le Mac
python -m registre.cli synth --n 3000
python -m registre.cli augment --src data/synth --out data/augmented_synth --per-page 2
python -m registre.cli augment --src data/clean --out data/augmented --per-page 6
python -m registre.cli finetune-dataset --src data/augmented data/augmented_synth
rsync -av data/finetune/ narval:$SCRATCH/registre_finetune/
```

## 2. Évaluation rapide (avant tout entraînement)

Sur le Mac, une page de grossesse prend plusieurs minutes. Sur un A100 avec vLLM et des appels
simultanés, l'évaluation de centaines de photos tient en une fraction d'heure (à mesurer).

```bash
# nœud de connexion, une seule fois
module load python/3.11 cuda arrow opencv
virtualenv --no-download $HOME/venvs/vllm && source $HOME/venvs/vllm/bin/activate
pip install vllm               # Qwen3-VL demande vLLM >= 0.11 ; vérifier : avail_wheels vllm
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir $SCRATCH/models/Qwen3-VL-8B-Instruct

# depuis le dossier registre-ocr copié sur Narval (avec data/dev, data/test_A... générés sur le Mac)
sbatch scripts/narval/eval.sbatch data/dev out/dev
# plafond de référence : le 32B (bf16 ≈ 64 Go -> 2 GPU A100 40 Go)
MODEL=Qwen3-VL-32B-Instruct GPUS=2 sbatch --gpus-per-node=a100:2 scripts/narval/eval.sbatch data/dev out/dev_32b
```

Le 8B sur Narval et le 8B sur le Mac doivent donner les mêmes lectures (même modèle, quantification
différente) : vérifier sur 2 ou 3 images avant de comparer quoi que ce soit.

## 3. Entraînement LoRA

```bash
sbatch scripts/narval/train.sbatch
```

Un GPU A100 40 Go suffit pour un LoRA sur un modèle de 7-8 milliards de paramètres, avec des
mosaïques de petite taille.

## 4. Évaluation : la seule qui compte

- **Ne jamais** évaluer sur des images vues à l'entraînement.
- Les patientes 9 et 10 sont réservées au test (`test.jsonl`). Le test final reste les pages remplies
  à la main et photographiées par l'équipe (jeu D), jamais vues à l'entraînement.
- Comparer le modèle de base et le modèle fine-tuné avec la même commande
  `python -m registre.cli evaluate`.

## 5. Revenir sur le Mac

1. Fusionner l'adaptateur LoRA :

   ```bash
   llamafactory-cli export merge.yaml
   ```

2. Convertir pour l'inférence locale. Deux options :
   - **MLX** : `mlx_vlm.convert --hf-path <modèle fusionné> -q`
   - **GGUF pour Ollama** : seulement si la version de llama.cpp prend en charge l'encodeur vision de
     ce modèle. Vérifier avant de se lancer.

3. Pointer `config.toml` vers le serveur local (LM Studio ou `mlx_vlm.server`).
