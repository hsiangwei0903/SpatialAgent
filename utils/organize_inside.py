import re
import json
from tqdm import tqdm

SPLIT = 'train'  # Change to 'val' if needed

def replace_masks_with_region(original_question: str) -> str:
    original_question = original_question.replace('<image>\n', '')

    token_pattern = re.compile(r'<mask>|[a-zA-Z]+|[^\s\w]')
    tokens = token_pattern.findall(original_question)

    mask_count = 0
    modified_tokens = []

    for i, token in enumerate(tokens):
        if token == '<mask>':
            replacement = f"[Region {mask_count}]"
            mask_count += 1
            modified_tokens.append(replacement)
        else:
            modified_tokens.append(token)

    modified_question = ' '.join(modified_tokens)

    region_count = modified_question.count('[Region')
    assert region_count == mask_count, f"Region count mismatch: {region_count} != {mask_count}"

    return modified_question

if __name__ == "__main__":
    
    input_path = f'../data/{SPLIT}/{SPLIT}.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = []

    data = [item for item in data if 'inside' in item['conversations'][1]['value']]

    for item in tqdm(data, desc="Processing JSON data"):
        image = item.get('image')
        rle = item.get('rle')
        question_id = item.get('id')
        rephrased_conversations = []
        conversation = item.get('conversations')
        question = item['conversations'][0]['value']
        response = item['conversations'][1]['value']
        rephrase_question = replace_masks_with_region(question)
        response_split = response.split('.')
        
        sentence_with_inside = [res for res in response_split if 'inside' in res]
        assert len(sentence_with_inside) == 1, f"Expected exactly one 'inside' in response, found {len(sentence_with_inside)}"
        
        inside_sentence = sentence_with_inside[0]
        assert 'inside the buffer region' in inside_sentence
        
        before, after = inside_sentence.split('inside the buffer region')
        all_ids = re.findall(r'\[Region (\d+)\]', rephrase_question)
        before_ids = re.findall(r'\[Region (\d+)\]', before)
        after_ids = re.findall(r'\[Region (\d+)\]', after)
        
        assert len(all_ids) == len(rle), f"Expected all_ids length {len(all_ids)} to match rle length {len(rle)}"
        assert len(after_ids) == 1, f"Expected exactly one region after 'inside the buffer region', found {len(after_ids)}"
        assert len(before_ids) != 0, f"Expected at least one region before 'inside the buffer region', found {len(before_ids)}"


        inside_ids = [int(id) for id in before_ids]
        buffer_id = int(after_ids[0])
        outside_ids = [int(id) for id in all_ids if int(id) not in inside_ids and int(id) != buffer_id]
        
        for id in inside_ids:
            processed_data.append(
                {
                    'image': image,
                    'inside': 1,
                    'buffer_rle': rle[buffer_id],
                    'obj_rle': rle[id], 
                }
            )
        
        for id in outside_ids:
            processed_data.append(
                {
                    'image': image,
                    'inside': 0,
                    'buffer_rle': rle[buffer_id],
                    'obj_rle': rle[id], 
                }
            )

    # number of inside
    inside_count = sum(1 for item in processed_data if item['inside'] == 1)
    outside_count = sum(1 for item in processed_data if item['inside'] == 0)

    print(f"Inside count: {inside_count}, Outside count: {outside_count}")
    print(f"Processed {len(processed_data)} items with inside/outside classification.")

    with open(f'../data/{SPLIT}/inside.json', 'w') as f:
        json.dump(processed_data, f, indent=4)
    

    print(f"Processing complete. Data saved to '../data/{SPLIT}/inside.json'.")