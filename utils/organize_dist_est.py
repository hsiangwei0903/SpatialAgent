import json
import re
from tqdm import tqdm

with open('data/train/train.json', 'r') as file:
    data = json.load(file)

data = [q for q in data if q['category'] == 'distance']

count = 0

refined_data = []
for item in tqdm(data):

    gpt_response = next(conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt')

    region_indices = sorted(set(int(x) for x in re.findall(r'\[Region (\d+)\]', gpt_response)))
    filtered_rle = [item['rle'][idx] for idx in region_indices if 0 <= idx < len(item['rle'])]

    refined_item = {
        'id': item['id'],
        'image': item['image'],
        'category': item['category'],
        'conversations': item['conversations'],
        'rle': filtered_rle,
        'normalized_answer': item['normalized_answer'],
        'freeform_answer': item['freeform_answer'],
    }

    assert len(refined_item['rle']) == 2, f"Expected 2 RLEs, got {len(refined_item['rle'])} for item ID {item['id']}"
    refined_data.append(refined_item)

with open('data/train/train_dist_est.json', 'w') as file:
    json.dump(refined_data, file, indent=4)


with open('data/val/val.json', 'r') as file:
    data = json.load(file)
    data = [item for item in data if item['category'] == 'distance']

count = 0

refined_data = []
for item in tqdm(data):

    gpt_response = next(conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt')

    region_indices = sorted(set(int(x) for x in re.findall(r'\[Region (\d+)\]', gpt_response)))
    filtered_rle = [item['rle'][idx] for idx in region_indices if 0 <= idx < len(item['rle'])]

    refined_item = {
        'id': item['id'],
        'image': item['image'],
        'category': item['category'],
        'conversations': item['conversations'],
        'rle': filtered_rle,
        'normalized_answer': item['normalized_answer'],
        'freeform_answer': item['freeform_answer'],
    }

    assert len(refined_item['rle']) == 2, f"Expected 2 RLEs, got {len(refined_item['rle'])} for item ID {item['id']}"
    refined_data.append(refined_item)

with open('data/val/val_dist_est.json', 'w') as file:
    json.dump(refined_data, file, indent=4)