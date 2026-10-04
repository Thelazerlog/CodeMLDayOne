# DayOne : une sage-femme, un téléphone et une IA

Prototype d'agent **type WhatsApp, hors ligne d'abord**, qui transforme la photo d'un registre maternel
papier en **dossier structuré**. Chaque champ porte un **statut** et une **confiance**. La sage-femme
**vérifie au lieu de ressaisir**, et les visites sont **reliées** au profil de la patiente par un code
aléatoire. Le registre papier reste l'outil de référence : le numérique s'y greffe.

Tout tourne en local. Aucune image ni aucune donnée ne quitte la machine, et le modèle de vision est
un modèle ouvert (Qwen3-VL-8B) exécuté sur place.

**🎬 Démo vidéo (2 min) : [YouTube](https://youtu.be/UasRawQHcDU)** (aussi dans le dépôt : [`Video_show_final.mp4`](Video_show_final.mp4)). On y voit une capture hors ligne, le retour du
réseau, la révision de champs incertains, une décision de correspondance de patiente, une coupure
pendant l'envoi et l'accès à l'image d'origine selon le rôle.

## Essayer en 2 minutes (sans GPU)

```bash
cd registre-ocr
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m app.server --lecteur enregistre      # puis ouvrir http://localhost:8000
```

`--lecteur enregistre` rejoue les **vraies lectures du modèle**, faites sur GPU (Narval) et stockées dans
`registre-ocr/data/demo/lectures.json`. Le jury n'a donc besoin ni de GPU ni de modèle pour voir le
vrai comportement. Le scénario de démo (capture hors ligne, retour du réseau, champ incertain,
correspondance de patiente) est décrit dans [`registre-ocr/app/README.md`](registre-ocr/app/README.md).

Autres modes :
- `--lecteur vlm` : lecture en direct avec Ollama (`qwen3-vl:8b-instruct`) ;
- `--lecteur simulation` : lecteur simulé à partir de la vérité terrain ;
- `--lecteur aucun` : IA indisponible, saisie manuelle complète.

## Résultats (évaluation champ par champ, reproductible)

| Jeu de test | Ce qu'il mesure | Champs | Exactitude | Erreurs silencieuses¹ | Révisions utiles² |
|---|---|---|---|---|---|
| **124 images officielles** du jeu de données | le jeu fourni (rendus nets du PDF) | 8 880 | **99,7 %** | **0,32 %** | – |
| **Pages synthétiques inédites** (40 pages, nouvelle graine) | valeurs, polices, chiffres et mots arabes jamais vus | 2 970 | **98,8 %** | **0,73 %** | – |
| **Photos dégradées simulées** (160) | flou, bougé, ombres, reflets, perspective, pli | 11 880 | **88,6 %** | **1,58 %** | **91 %** |
| … dégradation légère | | 5 940 | 99,2 % | 0,56 % | |
| … dégradation moyenne | | 5 940 | 78,0 % | 2,87 % | 92 % |
| **Vraies photos du carnet** (5, vérité terrain saisie par l'équipe) | autre mise en page, écriture cursive réelle | 177 | **48 %** | voir³ | 60 % |

1. **Erreur silencieuse** : champ que l'agent tranche seul (sans demander) et qui est faux. C'est la
   métrique de la consigne « un agent qui ne cache jamais ses doutes ». Les règles ajoutées cette nuit
   (chronologie des dates, écriture inattendue, accents perdus) l'ont fait passer de **2,6 % à 1,6 %**
   sur les photos dégradées.
2. **Révision utile** : quand l'agent demande une vérification, le champ était réellement faux.
3. Le vrai carnet n'a pas la même mise en page que le spécimen : on passe en **mode libre**, où le
   modèle transcrit la page et nous rangeons ses valeurs dans le schéma. **Aucune valeur n'y est
   présentée comme sûre** : tout est « à vérifier », avec une confirmation groupée. Les « erreurs
   silencieuses » mesurées ici sont des champs écrits que le modèle **n'a pas trouvés**, pas des
   valeurs fausses affirmées. Nos itérations de la nuit ont fait passer ce jeu de **17 % à 48 %**
   d'exactitude.

Les rapports complets (par type de champ, par page, courbe seuil → couverture) sont dans
[`registre-ocr/resultats/`](registre-ocr/resultats/). Pour recalculer sans GPU le chiffre des images
officielles :

```bash
cd registre-ocr && python -m registre.cli evaluate data/officiel --reader enregistre --out out/officiel
```

### Fine-tuning LoRA : essayé, mesuré, **non retenu**

LoRA (rang 16) sur Qwen3-VL-8B : 6 000 mosaïques issues de pages synthétiques et des patientes 1 à 7,
1 époque, 52 min sur un A100. La perte finale est de 0,012. L'évaluation se fait sur des données
jamais vues :

| Jeu | Modèle de base | Avec LoRA |
|---|---|---|
| Photos dégradées, patientes 9 et 10 (jamais vues) | 88,6 % / 1,6 % d'erreurs silencieuses (toutes patientes) | 86,2 % / **4,0 %** |
| Vraies photos du carnet | **48 %** | 13,6 % |

Le LoRA s'est trop spécialisé sur nos données synthétiques. Il fait davantage d'erreurs silencieuses
et perd sa capacité à lire une page entière (mode libre). **On garde le modèle de base.** Pour
progresser sur les vraies photos, il faut de vraies données annotées, pas plus de synthétique.

## Ce qui répond à chaque exigence

| Consigne | Où | Comment |
|---|---|---|
| 1. Schéma et statuts | `registre/schema.py`, `templates/*.json` | 8 pages et environ 600 champs déduits du registre. Statuts CONNU, INCONNU, NON_FOURNI, ILLISIBLE, NON_APPLICABLE et A_REVISER, avec confiance, raison du doute et provenance (image, zone, méthode). |
| 2. Pipeline d'extraction évalué | `registre/pipeline.py`, `registre/evaluate.py` | Qualité de la photo, alignement sur le gabarit, **masquage des identifiants avant lecture**, cases lues par OpenCV, texte lu par VLM local, fusion des signaux, règles de cohérence. |
| 3. Vérification conversationnelle | `app/conversation.py` | Bilan, puis questions par priorité : *Confirmer / Corriger / Reprendre la photo / Laisser vide*. Questions de suivi, saisie manuelle complète, sessions multipages. L'agent dit pourquoi il doute. |
| 4. Hors ligne (simulé) | `app/store.py`, `app/sync.py` | Stockage **chiffré** (Fernet, clé dérivée du PIN), file « En attente de traitement IA », reprise automatique au retour du réseau. Une coupure pendant une lecture ou un envoi compte comme un échec, avec réessai. |
| 5. Cycle de vie | `app/lifecycle.py` | CAPTURE, EN_ATTENTE_IA, TRAITE_IA, A_REVISER, VALIDE, PATIENTE_LIEE, ENREGISTRE, SYNCHRONISE, plus les états d'échec. Transitions vérifiées et journalisées. |
| 6. Liaison patiente | `app/linking.py` | Code aléatoire, tolérant aux confusions d'écriture (0/O, 1/I…). *[Patiente 1] [Patiente 2] [Aucune, créer] [Je ne sais pas]*. **Jamais de création automatique.** UUID interne. |
| 7. Multipage et renumérisation | `pipeline.merge_pages`, `linking.differences` | Les pages d'un registre forment un seul dossier. Une page rephotographiée garde la meilleure lecture. Une valeur qui change n'est jamais écrasée sans l'accord de la sage-femme. |
| 8. Image d'origine | `app/store.py`, `app/server.py` | Chiffrée, jamais modifiée (SHA-256), liée au dossier, à la date, à la sage-femme et au statut. **Accès selon le rôle**, chaque tentative est journalisée. |
| Bonus | | Contrôle qualité de la photo à la capture, agrégats anonymisés (TA, VIH, syphilis, hépatite B), robustesse aux chiffres et mots arabes (pages synthétiques). |

## Confidentialité

- **Aucun identifiant direct n'est lu ni stocké.** Nom, CIN, téléphone et adresse sont masqués sur
  l'image avant toute lecture. Le schéma du mode libre n'a pas de champ d'identité, et un filtre
  supprime en plus tout ce qui ressemble à un CIN ou un téléphone.
- L'identifiant patiente est un **UUID aléatoire**. Le code écrit sur le registre est stocké chiffré.
- **Pas d'API WhatsApp réelle, par choix.** Avec l'API Cloud de Meta, les messages sont déchiffrés sur
  les serveurs de Meta avant d'atteindre l'entreprise : le chiffrement de bout en bout s'arrêterait
  chez un tiers. Le prototype reproduit donc l'expérience WhatsApp (boutons de réponse rapide) en local.
- Seules des données synthétiques ont été envoyées sur Narval.

## Reproduire de zéro

**Prérequis** : macOS (Apple Silicon conseillé) ou Linux, Python ≥ 3.11, 16 Go de RAM et environ 10 Go
de disque pour le modèle. Un GPU NVIDIA est facultatif ; il ne sert qu'à aller plus vite.

### Le modèle : où il est, comment l'obtenir

Les poids du modèle **ne sont pas dans le dépôt** : 6 Go en version Ollama, 17 Go en bf16, alors que
GitHub limite un fichier à 100 Mo. Le modèle est **identifié exactement** et se télécharge en une
commande :

| Usage | Modèle | Commande | Taille |
|---|---|---|---|
| Portable (Ollama) | `qwen3-vl:8b-instruct` (empreinte `0533d74300e4`) | `ollama pull qwen3-vl:8b-instruct` | 6,1 Go |
| GPU (vLLM) | `Qwen/Qwen3-VL-8B-Instruct` (Hugging Face, bf16) | `hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct` | 17,5 Go |

Il n'y a aucun poids personnalisé à fournir : le fine-tuning LoRA n'a pas été retenu (voir plus haut).
Utilisez bien la variante **`-instruct`**. `qwen3-vl:8b` est la variante « thinking », lente et dont
les réponses sont tronquées.

### Étape 1 : code et environnement Python

```bash
git clone https://github.com/Thelazerlog/CodeMLDayOne.git
cd CodeMLDayOne/registre-ocr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                      # 30 tests, sans modèle ni GPU (environ 40 s)
```

### Étape 2 : sans modèle (vraies lectures enregistrées sur GPU)

```bash
# chiffres du README, à l'identique : 124 images officielles -> 99,7 %
python -m registre.cli evaluate data/officiel --reader enregistre --out out/officiel
# prototype conversationnel -> http://localhost:8000
python -m app.server --lecteur enregistre
```

### Étape 3 : avec le modèle en local (Ollama)

```bash
# installer Ollama
brew install ollama                                  # macOS
# curl -fsSL https://ollama.com/install.sh | sh      # Linux
ollama serve &                                       # serveur local (port 11434)
ollama pull qwen3-vl:8b-instruct                     # 6,1 Go

cp config.toml.example config.toml                   # pointe vers http://localhost:11434/v1
python -m registre.cli check-vlm                     # doit afficher « OK »

# lire un registre et produire le dossier JSON (out/dossier_<id>.json)
python -m registre.cli process data/demo/patient_02_*aug00*.jpg --code B4T7 --sage-femme SF-01

# évaluations champ par champ (rapport dans out/<nom>/report.md)
python -m registre.cli evaluate data/reel --reader vlm --out out/reel          # 5 vraies photos (environ 10 min)
python -m registre.cli evaluate data/demo --reader vlm --out out/demo          # 32 photos dégradées
python -m registre.cli evaluate data/officiel --reader vlm --out out/officiel  # 124 images (plusieurs heures sur Mac)

# prototype avec lecture en direct
python -m app.server --lecteur vlm
```

Ollama sert une version **quantifiée** du modèle (Q4) : les lectures peuvent différer légèrement des
nôtres, faites en bf16 sur GPU. L'étape 2 rejoue exactement ces dernières.

### Étape 4 : régénérer les données et la vérité terrain

```bash
python -m registre.cli build-templates               # gabarits + data/clean + data/gt depuis le PDF spécimen
python -m registre.cli augment --per-page 2 --out data/augmented       # 160 photos dégradées simulées
python -m registre.cli synth --n 40 --seed 777 --out data/synth_test   # pages synthétiques inédites
python -m registre.cli augment --src data/synth_test --out data/synth_test_aug --per-page 1 --seed 777
python -m registre.cli gt-csv data/reel              # vérité terrain des vraies photos (CSV lisibles -> JSON)
```

### Étape 5 : sur GPU (vLLM), pour aller vite et enregistrer les lectures

```bash
pip install "vllm>=0.11"
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
vllm serve models/Qwen3-VL-8B-Instruct --served-model-name registre --port 8000 \
     --max-model-len 8192 --limit-mm-per-prompt '{"image": 1}' &
```

Dans `config.toml`, utilisez ensuite :
`base_url = "http://localhost:8000/v1"`, `model = "registre"`, `backend = "openai"`, `parallel = 16`.

```bash
python -m registre.cli evaluate data/augmented --out out/aug             # environ 10 min sur un A100
python -m registre.cli record data/demo --out data/demo/lectures.json    # lectures rejouables sans GPU
```

Sur une grappe Slurm (nous avons utilisé Narval), `scripts/narval/vlm_job.sbatch` lance vLLM puis
n'importe quelle commande, avec ou sans adaptateur LoRA (variable `LORA`). Fine-tuning :
`prep_data.sbatch` (données, CPU), puis `train_nuit.sbatch` (LLaMA-Factory, environ 1 h sur A100). Voir
[`registre-ocr/scripts/narval/README.md`](registre-ocr/scripts/narval/README.md).

## Limites connues et suite

- **Vraies photos** : le carnet réel diffère du spécimen (typographie, espacement, pages en vis-à-vis).
  Le mode libre lit environ la moitié des champs. Une page de droite photographiée seule (sans les
  libellés des lignes) n'est pas lisible : l'agent **demande alors de photographier la double page**.
  La suite logique : un gabarit construit sur le vrai carnet, et une vérité terrain réelle plus large.
- **Fine-tuning** : un LoRA appris sur des pages synthétiques dégrade les résultats (voir plus haut).
- Formulaire : Ag HBs (hépatite B) au lieu de l'hépatite C du défi ; pas de température prénatale.
- Téléphone simulé dans un navigateur, serveur central simulé (SQLite), rôles sans authentification
  réelle. Une seule sage-femme à la fois dans la démo.

## Structure

```
data/              jeu de données fourni par les organisateurs (non modifié) : 124 images, 5 vraies photos, CSV
Video_show_final.mp4  vidéo de démonstration (2 min)
registre-ocr/
  registre/        pipeline d'extraction (schéma, alignement, lecture, fusion, règles, mode libre, évaluation)
  app/             prototype conversationnel, file hors ligne chiffrée, cycle de vie, liaison patiente
  templates/       gabarits des 8 pages (déduits du PDF spécimen)
  data/demo/       photos de démo + lectures réelles enregistrées (rejouables sans GPU)
  data/reel/       5 vraies photos + vérité terrain saisie par l'équipe (CSV lisibles)
  data/officiel/   liens vers les 124 images officielles + correspondance page -> patiente
  resultats/       rapports d'évaluation (Narval)
  scripts/narval/  jobs Slurm (vLLM, préparation des données, LoRA)
  tests/           30 tests sans modèle ni GPU
```
