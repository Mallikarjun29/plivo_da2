import json

def check_gold(path):
    print(f"--- Checking {path} ---")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            print(f"\nID: {obj['id']}")
            print(f"Text: '{text}'")
            
            for e in obj.get("entities", []):
                s, end, label = e["start"], e["end"], e["label"]
                span = text[s:end]
                print(f"  Label: {label:<12} | Indices: {s}-{end} | Extracted: '{span}'")

if __name__ == "__main__":
    check_gold("data/dev.jsonl")