# Registre maternel : de la photo au dossier structuré

Partie **traitement d'image et constitution du dossier** du défi DayOne. Le prototype conversationnel
(agent type WhatsApp, file hors ligne chiffrée, cycle de vie, liaison patiente) est dans
[`app/`](app/README.md) : `python -m app.server --lecteur simulation`, puis http://localhost:8000. Tout tourne en local : aucune image ni aucune donnée ne quitte la machine.

```
photo ─► qualité ─► type de page + alignement ─► masquage ─► lecture ─► statuts ─► dossier
         (flou,     (repères imprimés du          (nom, CIN,   (cases :     (confiance,   (multipage,
          lumière,   formulaire, SIFT + NCC)       tél...)      OpenCV ;     règles)       doublons,
          cadrage)                                              texte :                    alertes)
                                                                VLM local)
```

## Démarrage rapide (Mac M-series)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ollama tesseract tesseract-lang      # tesseract est facultatif (second avis sur les nombres)
ollama serve &                                    # serveur local
ollama pull qwen3-vl:8b-instruct                  # ≈ 6 Go ; surtout PAS « qwen3-vl:8b » (= variante thinking)
cp config.toml.example config.toml

python -m registre.cli build-templates            # gabarits + vérité terrain depuis le PDF
python -m registre.cli check-vlm                  # le modèle local répond-il ?
python -m registre.cli process data/clean/patient_01_*.png --code K7Q2 --sage-femme SF-01
python -m pytest -q                               # 30 tests, sans modèle ni GPU
```

`process` écrit `out/dossier_<id>.json`, plus les images redressées et masquées dans `out/masque/`.
La commande affiche aussi les champs à réviser, avec la raison de chaque doute.

## Ce que fait chaque étape

| Étape | Fichier | Idée clé |
|---|---|---|
| Gabarits + vérité terrain | `build_templates.py`, `pdf_layout.py` | Le PDF a une couche vectorielle. Libellés imprimés (Helvetica) = repères. Glyphes manuscrits = valeurs. Carrés de 8 pt = cases. Les zones sont déduites automatiquement des règles de tableau et des lignes « Libellé : ____ ». |
| Qualité | `quality.py` | Estime le **rayon de flou en pixels**, ramené à l'échelle du gabarit, dans 4 directions (le bougé est directionnel). Corrélation avec le flou réel : **≈ 0,75 à 0,85** selon le jeu. La variance du laplacien ne corrélait pas du tout. S'y ajoutent l'analyse par tuiles, l'exposition, les reflets et le cadrage, chacun avec un message lisible par la sage-femme. |
| Alignement | `align.py` | Aucun repère ajouté au papier. 1) SIFT entre la photo et le gabarit vierge. 2) Chaque libellé imprimé est recherché par corrélation (NCC) à plusieurs niveaux de flou. 3) Homographie finale. Donne aussi la part de chaque zone visible (« hors cadre »). |
| Type de page | `align.py` | Le gabarit qui retrouve le plus de repères gagne. Les pages jumelles (précoce/tardif) sont départagées sur les seuls repères qui diffèrent. |
| Masquage | `pipeline.py` | Les zones d'identifiants sont noircies **avant** toute lecture. Elles ne sont jamais transcrites, ni dans la sortie ni dans la vérité terrain. |
| Cases | `readers/ink.py` | Densité d'encre dans la case, sans modèle. |
| Texte | `readers/vlm.py` | Les zones contenant de l'encre sont empilées en **mosaïque numérotée** (une image par appel), avec un schéma JSON imposé. La confiance vient des logprobs si le serveur les fournit. Toute URL non locale et toute variante `-cloud` sont refusées. |
| Statuts | `fusion.py` | Croise plusieurs signaux : lecture, second lecteur, encre détectée, contraste des traits imprimés, format, visibilité, alignement, qualité. Un champ n'est CONNU que si ces signaux concordent. |
| Règles | `rules.py` | Bornes physiologiques, G/P/avortements, DPA = DDR + 280 j, SA cohérent avec les dates, mode d'accouchement, délais post-partum. Les règles **signalent** et ne corrigent jamais. Elles déduisent aussi les NON_APPLICABLE. |
| Dossier | `pipeline.py`, `schema.py` | Fusion multipage. Une page photographiée deux fois garde la lecture la plus sûre ; si deux lectures sûres divergent, conflit à réviser. Identifiant interne aléatoire. Provenance de chaque champ (image, zone, méthode). |
| Export | `export.py` | Une ligne par dossier, colonnes du CSV de 200 lignes. Seuls les champs CONNU ou validés alimentent l'export. |

### Mode libre (mise en page inconnue) et questions de suivi

- **Mode libre** (`freeform.py`) : si aucun gabarit ne correspond (vrai carnet, autre édition, page très
  abîmée), le modèle local classe la page parmi nos 8 types puis range ce qu'il lit dans **notre** schéma
  (mêmes clés : lignes × colonnes des tableaux, champs simples, cases). Le dossier garde la même structure.
  Sans gabarit, rien n'est vérifié géométriquement : tout est `A_REVISER` et fait l'objet d'une
  confirmation groupée. Aucun champ d'identité dans le schéma de sortie, filtre CIN/téléphone en plus.
- **Questions de suivi** (`questions.py`, champ `questions` du dossier) : reprendre la photo, type de page
  à confirmer, incohérences, valeurs douteuses ou aberrantes (bornes + unités, ex. « 324 m » pour un
  périmètre crânien), champs illisibles, champs importants non trouvés, pages manquantes. Triées par
  priorité, avec les options à proposer (Confirmer / Corriger / Reprendre la photo / Saisir).

### Statuts

| Statut | Signification |
|---|---|
| `CONNU` | Lu, et plusieurs signaux concordent |
| `NON_FOURNI` | Zone vide (confirmée par l'image) ou tiret « — » |
| `INCONNU` | « ? », « NSP » écrit sur le registre |
| `ILLISIBLE` | Encre présente, mais le modèle avoue ne pas pouvoir lire |
| `NON_APPLICABLE` | Déduit par une règle (indication de césarienne après voie basse, RAI si Rh+, cicatrice sans césarienne) |
| `A_REVISER` | Doute, avec la raison en clair. Exemples : hors cadre, reflet, vide non confirmé, format invalide, désaccord entre lecteurs ou entre photos, règle violée |

## Mesures obtenues ici (sans GPU, sur 48 photos augmentées des patientes 1-2)

**Alignement**
- Erreur médiane **≈ 0,6 px** sur les coins de page, aux trois niveaux de dégradation.
- Type de page reconnu : léger 16/16, moyen 16/16, fort 14/16.

**Cases à cocher** : exactitude **100 %** (léger), **100 %** (moyen), **99,7 %** (fort), sur 1 116 cases visibles.

**Plomberie de bout en bout**, avec un lecteur SIMULÉ (vérité terrain bruitée à 8 %). Ces chiffres ne
mesurent **pas** la lecture manuscrite : ils vérifient que les statuts protègent contre les erreurs
silencieuses.
- Erreurs silencieuses : **0,2 %**
- Révisions : 2 % des champs (photos légères), 15 % (moyennes), 47 % (fortes)

Le vrai chiffre viendra de
`python -m registre.cli evaluate data/augmented --reader vlm --tesseract` sur ton Mac. Le rapport sort
dans `out/eval/report.md`, avec la courbe seuil → couverture / erreurs silencieuses.

## Données synthétiques et augmentation

```bash
python -m registre.cli augment --per-page 6        # 10 dossiers -> 480 photos dégradées, géométrie exacte
python -m registre.cli synth --n 400               # pages remplies avec d'AUTRES polices, arabe, « ? », gribouillis
python -m registre.cli augment --src data/synth --out data/augmented_synth --per-page 2
python -m registre.cli finetune-dataset --src data/augmented data/augmented_synth   # pour Narval
python -m registre.cli calibrate-quality           # recalcule le seuil de flou
```

L'augmentation simule :
- rotation (y compris 90°) et perspective ;
- page petite, décentrée ou **coupée** ;
- fond de table ;
- ombre portée, reflet, faible lumière, dominante de couleur ;
- flou de mise au point et de bougé, bruit, compression JPEG, pli du papier.

Chaque photo garde son homographie exacte : on peut ainsi mesurer l'alignement au pixel près.

Le fine-tuning se fait sur Narval : voir `scripts/narval/README.md`.

## Où intervenir (dans cet ordre)

1. **Télécharger le Drive** dans `data/drive/` et lire le manifeste. Deux questions à trancher :
   - Les 129 images suivent-elles la mise en page du PDF ?
   - Comment est fournie leur vérité terrain ?

   Ensuite, écrire un petit adaptateur vers `data/gt/patient_XX.json` (même format que celui généré), ou
   un CSV `image,patient,page_type` puis `python -m registre.cli sidecars mapping.csv`.
2. **Si des pages ont une autre mise en page** (versions arabe ou anglaise du registre) : il faut un
   gabarit par mise en page.
   - Si un PDF de cette mise en page existe, `build_templates.py` le traite tel quel.
   - Sinon, prendre une photo propre et nette et annoter ses zones dans un JSON au même format que
     `templates/*.json`. Un outil comme Label Studio le fait vite.
3. **Relire les 8 images `templates/*_overlay.png`** et corriger les noms de clés si besoin
   (`KEY_RENAMES` dans `build_templates.py`). Codes couleur :
   - vert : repères ;
   - orange : champs texte ;
   - bleu : cases ;
   - rouge : zones masquées.
4. **Lancer l'évaluation avec le VLM** sur ton Mac, puis régler `FusionParams.seuil_connu` (`fusion.py`)
   avec la courbe du rapport. Cible : moins de 1 % d'erreurs silencieuses.
5. **Recalibrer** `calibrate-quality` et le seuil des cases (`CHECK_THRESHOLD`, `readers/ink.py`) sur
   un plus gros jeu d'augmentations, puis sur les vraies photos.
6. **Décider** si « vu par » et « examen fait par » (noms du personnel) sont stockés tels quels, réduits
   au rôle (« Sage-femme », « Dr »), ou masqués.
7. Brancher la couche WhatsApp. Ce qu'elle trouve dans le dossier :
   - `Dossier.champs_a_reviser()` : les champs à réviser et leurs raisons ;
   - `alertes` : les incohérences ;
   - `capture.qualite.raisons` : les motifs de reprise de photo ;
   - `type_page_a_confirmer` : la question « précoce ou tardive ? ».

## Limites connues

- La lecture manuscrite réelle **n'a pas été testée ici** : l'environnement de développement n'avait
  ni GPU ni accès aux modèles. Le protocole est testé contre un faux serveur, pas la qualité de lecture.
- Gabarits construits à partir du seul PDF spécimen. Une autre édition du registre demande un nouveau
  gabarit.
- Le formulaire contient Ag HBs (hépatite B), pas l'hépatite C du défi : la colonne reste vide.
- Il n'y a pas de ligne de température prénatale sur ce formulaire.
- Le seuil de flou est calibré sur seulement 48 photos : à refaire sur un jeu plus grand.
- Chiffrement, rôles d'accès à l'image d'origine et file hors ligne : voir [`app/README.md`](app/README.md).
- Incohérences présentes dans les données synthétiques, **volontaires ou non**, et signalées par les
  règles sans être corrigées :
  - G3 P1 sans avortement ;
  - « avortement » à 34 SA ;
  - utérus cicatriciel sans césarienne antérieure ;
  - certaines patientes (3, 4, 8, 9) ont des glyphes manquants (é, —) dans la police ; la vérité terrain
    les restaure.
