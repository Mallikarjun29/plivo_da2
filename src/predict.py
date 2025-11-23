import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
            
        label = ID2LABEL.get(int(lid), "O")
        
        # If we encounter 'O' or a different label, close the current span
        if label == "O" or (current_label and label != f"I-{current_label}" and label != f"B-{current_label}"):
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
                current_start = None
                current_end = None

        if label == "O":
            continue

        prefix, ent_type = label.split("-", 1)
        
        if prefix == "B":
            if current_label is not None:
                 spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
            
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                # Treat mismatched I-tag as a new start
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    # --- FIX: Whitespace Trimming ---
    final_spans = []
    for start, end, label in spans:
        if start >= end: continue
        
        # Trim leading spaces
        while start < end and text[start].isspace():
            start += 1
        # Trim trailing spaces
        while end > start and text[end-1].isspace():
            end -= 1
            
        if start < end:
            final_spans.append((start, end, label))

    return final_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    
    print(f"Loading model from: {args.model_dir}")
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    
    # --- FIX: Dynamic Quantization for Latency ---
    if args.device == "cpu":
        print("Optimizing model with Dynamic Quantization for CPU...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            
            # Move inputs to device
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
