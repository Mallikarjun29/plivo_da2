import json
from typing import List, Dict, Any
from torch.utils.data import Dataset


class PIIDataset(Dataset):
    def __init__(self, path: str, tokenizer, label_list: List[str], max_length: int = 256, is_train: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length
        self.is_train = is_train

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj["text"]
                entities = obj.get("entities", [])

                # Create a character-level map of labels
                char_tags = ["O"] * len(text)
                for e in entities:
                    s, e_idx, lab = e["start"], e["end"], e["label"]
                    if s < 0 or e_idx > len(text) or s >= e_idx:
                        continue
                    
                    char_tags[s] = f"B-{lab}"
                    for i in range(s + 1, e_idx):
                        char_tags[i] = f"I-{lab}"

                # Tokenize
                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                )
                offsets = enc["offset_mapping"]
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

                bio_tags = []
                for idx, (start, end) in enumerate(offsets):
                    # Special tokens (CLS, SEP, PAD) have (0, 0) offset usually, or are outside text
                    if start == end or (start == 0 and end == 0 and idx != 0): 
                        # Note: idx!=0 check handles the case where the first token is valid text starting at 0
                        # But usually CLS is (0,0). Let's rely on the fact that CLS/SEP are not in char_tags range usually
                        # or just default to 'O' for special tokens.
                        bio_tags.append("O")
                        continue
                    
                    # IMPROVED LOGIC: Scan the whole token span
                    # If the token contains a B- tag, label it B-.
                    # If it contains only I- tags, label it I-.
                    token_span_tags = char_tags[start:end] if start < len(char_tags) else []
                    
                    if not token_span_tags:
                        bio_tags.append("O")
                        continue

                    # Check for B- tags first
                    b_tags = [t for t in token_span_tags if t.startswith("B-")]
                    if b_tags:
                        # Use the first B- tag found (usually only one)
                        bio_tags.append(b_tags[0])
                    else:
                        # Check for I- tags
                        i_tags = [t for t in token_span_tags if t.startswith("I-")]
                        if i_tags:
                            bio_tags.append(i_tags[0])
                        else:
                            bio_tags.append("O")

                # Ensure lengths match
                if len(bio_tags) != len(input_ids):
                    bio_tags = bio_tags[:len(input_ids)]
                    while len(bio_tags) < len(input_ids):
                        bio_tags.append("O")

                label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]

                final_labels = []
                for i, (start, end) in enumerate(offsets):
                    if start == end: # Special token
                        final_labels.append(-100)
                    else:
                        final_labels.append(label_ids[i])

                self.items.append(
                    {
                        "id": obj["id"],
                        "text": text,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": final_labels,
                        "offset_mapping": offsets,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_batch(batch, pad_token_id: int, label_pad_id: int = -100):
    input_ids_list = [x["input_ids"] for x in batch]
    attention_list = [x["attention_mask"] for x in batch]
    labels_list = [x["labels"] for x in batch]

    max_len = max(len(ids) for ids in input_ids_list)

    def pad(seq, pad_value, max_len):
        return seq + [pad_value] * (max_len - len(seq))

    input_ids = [pad(ids, pad_token_id, max_len) for ids in input_ids_list]
    attention_mask = [pad(am, 0, max_len) for am in attention_list]
    labels = [pad(lab, label_pad_id, max_len) for lab in labels_list]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": [x["offset_mapping"] for x in batch],
    }
    return out
