import json
import random
import os

TRAIN_FILE = "data/train_synthetic.jsonl"
DEV_FILE = "data/dev_synthetic.jsonl"
NUM_TRAIN = 1000
NUM_DEV = 200

# --- Data Pools ---
NAMES = ["ramesh", "suresh", "priya", "anita", "john", "doe", "vikram", "rahul", "sneha", "arun", "deepak", "meera"]
SURNAMES = ["sharma", "verma", "gupta", "patel", "singh", "kumar", "reddy", "rao", "nair", "iyer", "malik", "khan"]
CITIES = ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "jaipur", "ahmedabad", "surat"]
LOCATIONS = ["mg road", "indira nagar", "connaught place", "marine drive", "airport", "central station", "whitefield", "andheri"]
DOMAINS = ["gmail", "yahoo", "outlook", "hotmail", "example"]

DIGIT_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", 
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"
}

# --- Generators ---
def get_person():
    return f"{random.choice(NAMES)} {random.choice(SURNAMES)}"

def get_phone():
    # Phone is strictly 10 digits
    nums = [str(random.randint(0, 9)) for _ in range(10)]
    if random.random() < 0.5:
        return "".join(nums)
    else:
        return " ".join([DIGIT_MAP[n] for n in nums])

def get_card():
    # Card is strictly 16 digits, usually groups of 4
    groups = []
    for _ in range(4):
        groups.append("".join([str(random.randint(0, 9)) for _ in range(4)]))
    return " ".join(groups)

def get_email():
    name = f"{random.choice(NAMES)} {random.choice(SURNAMES)}"
    n = name.replace(" ", " dot ")
    d = random.choice(DOMAINS)
    return f"{n} at {d} dot com"

def get_date():
    d = str(random.randint(1, 31)).zfill(2)
    m = str(random.randint(1, 12)).zfill(2)
    y = str(random.randint(2020, 2025))
    return f"{d} {m} {y}"

def get_city():
    return random.choice(CITIES)

def get_location():
    return random.choice(LOCATIONS)

# --- Templates ---
TEMPLATES = [
    ("my name is {0}", ["PERSON_NAME"]),
    ("i am {0} calling from {1}", ["PERSON_NAME", "CITY"]),
    ("contact me at {0}", ["PHONE"]),
    ("my number is {0} and email is {1}", ["PHONE", "EMAIL"]),
    ("credit card is {0}", ["CREDIT_CARD"]),
    ("pay using card {0} expiry {1}", ["CREDIT_CARD", "DATE"]),
    ("i live in {0} near {1}", ["CITY", "LOCATION"]),
    ("traveling to {0} on {1}", ["CITY", "DATE"]),
    ("email id is {0}", ["EMAIL"]),
    ("send details to {0}", ["EMAIL"]),
    ("my phone is {0}", ["PHONE"]),
    ("card number {0}", ["CREDIT_CARD"]),
    ("this is {0} from {1}", ["PERSON_NAME", "CITY"]),
    ("meeting on {0} at {1}", ["DATE", "LOCATION"]),
    
    # --- BOOSTED DATE TEMPLATES ---
    # Duplicating these to force the model to learn them
    ("i will travel on {0}", ["DATE"]),
    ("i will travel on {0}", ["DATE"]), 
    ("traveling on {0}", ["DATE"]),
    ("departure on {0}", ["DATE"]),
    ("date is {0}", ["DATE"]),
    ("on {0}", ["DATE"]), # Very short context to force attention to the format
    
    # Existing ones
    ("my card number is {0}", ["CREDIT_CARD"]),
    ("please charge my card {0}", ["CREDIT_CARD"]),
    ("the card is {0}", ["CREDIT_CARD"]),
    ("date of birth {0}", ["DATE"]),
    ("schedule for {0}", ["DATE"]),
]

def generate_dataset(filename, num_samples, start_id=0):
    data = []
    for i in range(num_samples):
        template, ent_types = random.choice(TEMPLATES)
        
        values = []
        for et in ent_types:
            if et == "PERSON_NAME": values.append(get_person())
            elif et == "PHONE": values.append(get_phone())
            elif et == "EMAIL": values.append(get_email())
            elif et == "CREDIT_CARD": values.append(get_card())
            elif et == "DATE": values.append(get_date())
            elif et == "CITY": values.append(get_city())
            elif et == "LOCATION": values.append(get_location())
        
        text = template
        for idx, val in enumerate(values):
            text = text.replace(f"{{{idx}}}", val, 1)
            
        entities = []
        search_start = 0
        for idx, (val, label) in enumerate(zip(values, ent_types)):
            start = text.find(val, search_start)
            end = start + len(val)
            entities.append({"start": start, "end": end, "label": label})
            search_start = end 
            
        data.append({
            "id": f"syn_{start_id + i:04d}",
            "text": text,
            "entities": entities
        })
        
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {num_samples} samples to {filename}")

if __name__ == "__main__":
    generate_dataset(TRAIN_FILE, NUM_TRAIN, start_id=0)
    generate_dataset(DEV_FILE, NUM_DEV, start_id=NUM_TRAIN)