import json

def evaluer_exactitude(predictions_json: str, ground_truth_json: str):
    with open(predictions_json, 'r') as f:
        preds = json.load(f)
    with open(ground_truth_json, 'r') as f:
        truths = json.load(f)
        
    total_champs = len(truths.keys()) if isinstance(truths, dict) else 0
    
    print("--- Rapport d'Évaluation Rapide ---")
    print(f"Total champs évalués : {total_champs}")
    print(f"Précision Valeur : {0.0}%")
    print(f"Précision Statut : {0.0}%")
    print("-----------------------------------")

if __name__ == '__main__':
    # evaluer_exactitude('preds.json', 'truth.json')
    pass