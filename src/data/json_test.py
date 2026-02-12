import json

# Path to your training JSON
json_path = "data/spoken_train-v1.1.json"

with open(json_path, 'r') as f:
    data = json.load(f)

print("--- JSON Structure Check ---")
article = data['data'][0]
print(f"First Article Title: {article['title']}")

paragraph = article['paragraphs'][0]
print(f"First Paragraph Context (start): {paragraph['context'][:50]}...")
print(f"First QA ID: {paragraph['qas'][0]['id']}")

print("\n--- Checking for 'id' matching ---")
# Does the JSON have the 0_0_0 style IDs?
print(f"ID of 1st QA: {paragraph['qas'][0]['id']}")