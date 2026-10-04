# Prototype conversationnel : agent type WhatsApp, hors ligne d'abord

Couche « téléphone » branchée sur le pipeline d'extraction (`registre/`). Elle couvre le flux de
vérification, la file hors ligne chiffrée, le cycle de vie des enregistrements et la liaison patiente.

## Lancer la démo

```bash
cd registre-ocr
pip install -r requirements.txt
python -m app.server --lecteur simulation        # puis ouvrir http://localhost:8000
```

| `--lecteur` | Usage |
|---|---|
| `enregistre` | **Vraies lectures du modèle** (Qwen3-VL-8B sur GPU Narval), enregistrées dans `data/demo/lectures.json` et rejouées sans GPU. C'est le mode conseillé pour le jury. Une photo non enregistrée part en saisie manuelle. |
| `simulation` | Démo rapide. Le lecteur renvoie la vérité terrain **bruitée à 8 %** : le bandeau l'indique. Montre le parcours, ne mesure pas la lecture. |
| `vlm` | Vraie lecture par le modèle local (Ollama, `qwen3-vl:8b-instruct`). Plusieurs minutes par page sur Mac. |
| `aucun` | IA indisponible : tout passe en saisie manuelle. |

Les photos de démo sont dans `data/demo/` (patientes 1 et 2, dégradation légère et moyenne) et dans
`../data/Paper Registry/` (vraies photos). D'autres photos dégradées se génèrent avec
`python -m registre.cli augment --per-page 2 --patients 1 2 --out data/augmented`.

Options : `--pin` (code PIN de l'appareil, d'où est dérivée la clé), `--racine` (données du téléphone
et du serveur simulés, par défaut `out/app`), `--sage-femme`, `--port`.

## Scénario de démo (≈ 3 min)

1. **Capture hors ligne** : panneau « Réseau » → *Hors ligne*. *📷 Nouveau registre* → code `B4T7` →
   choisir `augmented / patient_02_…` → *Toutes les pages de cette patiente*. Contrôle qualité sur le
   téléphone (photo floue, reflet, page coupée → *Reprendre* / *Garder*). *J'ai terminé* → « En attente
   de traitement IA », chiffré sur le téléphone.
2. **Retour de la connexion** : *En ligne*. La file part seule, sans aucune action :
   `EN_ATTENTE_IA → TRAITE_IA → A_REVISER` (colonne Historique).
3. **Révision d'un champ incertain** : l'agent présente le bilan (sûrs / vides / à vérifier / sans
   objet), puis pose ses questions par priorité, avec la raison du doute et la confiance :
   *Confirmer / Corriger / Reprendre la photo / Laisser vide / Passer*. Une correction est validée selon
   le type du champ (date, TA, nombre…).
4. **Décision de correspondance** : *Aucune, créer* pour une première visite. Recommencer avec le code
   `B4T8` (une lettre d'écart) et une autre photo de la même patiente : l'agent propose
   *[Patiente 1] [Aucune, créer] [Je ne sais pas]* avec ses raisons (code proche, même DDR, même âge).
   Il ne crée jamais de patiente tout seul.
5. **Coupure pendant l'envoi** : passer *Hors ligne* juste après l'enregistrement → `ECHEC_SYNC`, puis
   *En ligne* → `SYNCHRONISE`, sans doublon côté serveur.
6. **Image d'origine** : panneau « accès selon le rôle ». La sage-femme auteure et le superviseur y
   ont accès, une autre sage-femme et l'analyste non. Chaque tentative est journalisée.

## Cycle de vie (`app/lifecycle.py`)

```
CAPTURE ─► EN_ATTENTE_IA ─► TRAITE_IA ─► A_REVISER ─► VALIDE ─► PATIENTE_LIEE ─► ENREGISTRE ─► SYNCHRONISE
                │  ▲              └──── (rien à vérifier) ──►┘
                ▼  │ réessai auto
          ECHEC_TRAITEMENT ─(3 échecs)─► REVISION_MANUELLE_REQUISE ─► VALIDE (saisie manuelle)
ENREGISTRE ─► ECHEC_SYNC ─► ENREGISTRE (réessai au retour du réseau)
CAPTURE ─► DOUBLON_SUSPECT ─► EN_ATTENTE_IA (« ce n'est pas un doublon ») | ANNULE
A_REVISER ─► EN_ATTENTE_IA (photo reprise : ce qui est déjà validé est conservé)
SYNCHRONISE ─► A_REVISER (renumérisation)
```

Toute transition est vérifiée et journalisée (date, raison). Une transition non prévue lève une
erreur : on ne peut ni sauter la révision, ni envoyer un dossier sans patiente.

## Choix de conception

- **Écrire d'abord, traiter ensuite.** La photo est chiffrée et enregistrée avant tout autre calcul :
  couper l'application ou le réseau ne perd rien.
- **Hors ligne = machine à états + travailleur.** Le travailleur (`app/sync.py`) vide la file dès que
  le réseau revient. Une coupure *pendant* une lecture ou un envoi compte comme un échec, avec réessai
  automatique. L'envoi est idempotent : la clé est l'id de l'enregistrement.
- **Chiffrement au repos** (`app/store.py`) : Fernet (AES + HMAC), clé dérivée du PIN (PBKDF2,
  390 000 itérations). En clair dans la base : uniquement l'id, l'état et les dates, utiles à la file.
  Le code patiente lui-même est chiffré.
- **Liaison** (`app/linking.py`) : correspondance par code, tolérante aux confusions d'écriture
  (0/O/Q/D, 1/I/L, 5/S, 2/Z, 8/B) et à une lettre d'écart. Le profil (âge, DDR) renforce ou affaiblit
  la suggestion. L'identifiant interne est un UUID aléatoire.
- **Renumérisation** : le même registre est rephotographié à chaque visite. Les nouvelles valeurs
  s'ajoutent. Une valeur qui change n'est jamais écrasée sans le choix de la sage-femme.
- **L'agent ne cache pas ses doutes** : *Passer* ou *Vérifier le reste plus tard* laissent le champ
  « à vérifier » dans le dossier. Seuls les champs sûrs ou validés alimentent les agrégats.
- **Saisie manuelle complète** : par page, et par visite pour la grossesse. Elle est disponible à tout
  moment et proposée automatiquement après 3 échecs de lecture.

## Limites connues

- Téléphone simulé dans un navigateur : sur un vrai appareil, la clé serait dans le Keystore Android ou
  la Keychain iOS, et la file tournerait en tâche de fond (WorkManager).
- Pas de connexion à l'API WhatsApp Business : l'interface reproduit le fonctionnement par boutons de
  réponse rapide.
- Une seule conversation, donc une seule sage-femme à la fois dans la démo.
- Le serveur central est simulé (SQLite local). Les rôles sont choisis dans le panneau, sans
  authentification réelle.
- L'hépatite C n'existe pas sur ce formulaire (Ag HBs = hépatite B).
