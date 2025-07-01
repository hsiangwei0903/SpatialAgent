import re
import json
from tqdm import tqdm
from google import genai

client = genai.Client(api_key='')  # Replace with your actual API key

prompt = open('agent/prompt/rephrase.txt', 'r').read()

def verify(original_question, rephrased_question: str) -> bool:
    original_count = original_question.count('<') + original_question.count('>')
    rephrased_count = rephrased_question.count('<') + rephrased_question.count('>')
    return original_count == rephrased_count

def rephrase_question(question: str) -> str:
    question = question.replace('<image>\n', '')
    input_text = prompt.replace('<input>', question)
    response = client.models.generate_content(
            model="gemini-2.5-pro-preview-06-05",
            contents=(
                input_text
            ),
            config=genai.types.GenerateContentConfig(
                thinking_config=genai.types.ThinkingConfig(thinking_budget=128)
            )
        )
    return response.text.strip()

def replace_masks_with_objects(original_question: str) -> str:
    original_question = original_question.replace('<image>\n', '')

    object_keywords = ['shelf', 'transporter', 'pallet', 'buffer']

    # Tokenize while keeping <mask> as a single token, words, and punctuation
    token_pattern = re.compile(r'<mask>|[a-zA-Z]+|[^\s\w]')
    tokens = token_pattern.findall(original_question)

    object_counters = {obj: 0 for obj in object_keywords}
    modified_tokens = []
    last_object = None
    last_object_updated = False

    for i, token in enumerate(tokens):
        if token == '<mask>':
            
            if not last_object_updated:
                rephrase = rephrase_question(original_question)
                print(f"Original question: {original_question}")
                print(f"Rephrased question: {rephrase}")
                if not verify(original_question, rephrase):
                    import pdb; pdb.set_trace()
                return rephrase   

            replacement = f"<{last_object}_{object_counters[last_object]}>"
            object_counters[last_object] += 1
            modified_tokens.append(replacement)
        else:
            if tokens[i - 1] == '<mask>':
                last_object_updated = False
            if token.lower() == 'shelves':
                token_lower = 'shelf'
            else:
                token_lower = token.lower().rstrip('s')
            if token_lower in object_keywords:
                last_object = token_lower
                last_object_updated = True
            modified_tokens.append(token)

    # Rebuild the sentence with appropriate spacing
    modified_question = ''
    prev_token = ''
    for token in modified_tokens:
        if prev_token:
            if (re.match(r'[a-zA-Z0-9>]', prev_token) and re.match(r'[a-zA-Z0-9<]', token)):
                modified_question += ' '
            elif re.match(r'[>)]', prev_token) and re.match(r'[A-Za-z<]', token):
                modified_question += ' '
            elif re.match(r'[A-Za-z0-9]', prev_token) and re.match(r'[<]', token):
                modified_question += ' '
        modified_question += token
        prev_token = token

    modified_question = ' '.join(modified_tokens)

    return modified_question


if __name__ == "__main__":
    
    test = True

    if test:
        input_path = 'data/test/test.json'
        output_path = 'data/test/rephrased_test.json'
    else:
        input_path = 'data/val/val.json'
        output_path = 'data/val/rephrased_val.json'

    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    
    for item in tqdm(data, desc="Processing JSON data"):
        question_id = item.get('id')
        rephrased_conversations = []
        for conversation in item.get('conversations', []):
            if conversation.get('from') == 'human':
                original_question = conversation.get('value')
                if original_question:
                    rephrased_question = replace_masks_with_objects(original_question)
                    rephrased_conversations.append({
                        "from": "human",
                        "value": rephrased_question
                    })
                else:
                    rephrased_conversations.append(conversation)
            else:
                rephrased_conversations.append(conversation)
        item['rephrase_conversations'] = rephrased_conversations
        processed_data.append(item)

    
    with open(output_path, 'w') as out_file:
        json.dump(processed_data, out_file, indent=4)
