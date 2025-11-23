import json

def debug_preds(gold_path, pred_path):
    with open(gold_path) as f:
        gold_data = {json.loads(line)["id"]: json.loads(line) for line in f}
    
    with open(pred_path) as f:
        pred_data = json.load(f)
        
    for uid, preds in pred_data.items():
        gold = gold_data.get(uid)
        if not gold: continue
        
        text = gold["text"]
        print(f"\nID: {uid}")
        print(f"Text: {text}")
        
        print("  GOLD:")
        for e in gold["entities"]:
            print(f"    {e['label']:<12} | {e['start']}-{e['end']} | '{text[e['start']:e['end']]}'")
            
        print("  PRED:")
        for e in preds:
            print(f"    {e['label']:<12} | {e['start']}-{e['end']} | '{text[e['start']:e['end']]}'")

if __name__ == "__main__":
    debug_preds("data/dev.jsonl", "out/dev_pred.json")