import re
import json
import os
import ast
from tools import tools_api
from google import genai
from argparse import ArgumentParser
from tqdm import tqdm
from mask import parse_masks_from_conversation

def parse_args():
    parser = ArgumentParser(description="Agent for answering questions with inside model.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud Project ID')
    parser.add_argument('--location', type=str, default='global', help='Location for the Vertex AI resources')
    parser.add_argument('--think_mode', action='store_true', help='Enable think mode for the agent')
    parser.add_argument('--output_path', type=str, default='../output/test.json', help='Path to save the results')
    return parser.parse_args()

class Agent:
    def __init__(self, project_id, location, tools_api, input, think_mode=False):
        self.tools_api = tools_api
        self.client = genai.Client(vertexai=True, project=project_id, location=location)
        
        if think_mode:
            self.chat = self.client.chats.create(
                model="gemini-2.5-flash-preview-05-20",
                config=genai.types.GenerateContentConfig(
                    thinking_config=genai.types.ThinkingConfig(thinking_budget=128), temperature=0.2)
                )
        else:
            self.chat = self.client.chats.create(
                model="gemini-2.5-flash-preview-05-20",
                config=genai.types.GenerateContentConfig(temperature=0.2)
            )

        self.messages = []
        self.conversation = []
        self.prompt_preamble = open('prompt/agent_example.txt', 'r').read()
        self.answer_preamble = open('prompt/answer.txt', 'r').read()
        self.input = input
        self.masks = None
        self.question = None

    def set_masks(self):
        conversation = self.input['rephrase_conversations'][0]['value']
        rle_data = self.input['rle']
        self.masks = parse_masks_from_conversation(conversation, rle_data)
        self.tools_api.update_masks(self.masks)
        if not self.masks:
            raise ValueError("No valid masks found in the conversation.")
    
    def format_answer(self):
        answer = self.chat.send_message(self.answer_preamble)
        answer = answer.text.strip()
        if answer in self.masks:
            return self.masks[answer].region_id
        else:
            return answer

    def set_question(self):
        self.messages = []
        self.question = self.input['rephrase_conversations'][0]['value']
        full_prompt = self.prompt_preamble.replace("<question>", self.question)
        self.messages.append({"role": "user", "content": full_prompt})
        self.tools_api.update_image('../data/test/images/' + self.input['image'])
        return self._conversation_loop()

    def _conversation_loop(self, budget=10):
        usage = 0
        execute_flag = False
        while usage < budget:
            usage += 1
            # Compose the prompt from the conversation history
            latest_message = self.messages[-1]["content"]
            response = self.chat.send_message(latest_message)
            assistant_text = response.text.strip()
            self.messages.append({"role": "assistant", "content": assistant_text})

            # Check for <execute> command
            execute_match = re.search(r"<execute>(.*?)</execute>", assistant_text, re.DOTALL)
            if execute_match:
                execute_flag = True
                command = execute_match.group(1).strip()
                result = self._execute_function(command)
                # Send the result back to Gemini
                self.messages.append({"role": "user", "content": f"{result}"})
                continue  # Continue the loop

            # Check for <answer> tag
            answer_match = re.search(r"<answer>(.*?)</answer>", assistant_text, re.DOTALL)
            if answer_match and execute_flag:
                final_answer = answer_match.group(1).strip()
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            print("No valid action found. Ending interaction.")
            raise ValueError("No valid action found in the assistant's response.")

    def _execute_function(self, command):
        """Parses and executes a function call from Gemini."""
        command = command.strip().replace('\\n', '').replace('\n', '').replace('\r', '').replace('<', '').replace('>', '')
        match = re.match(r"(\w+)\s*\((.*)\)", command)
        if not match:
            raise ValueError(f"Invalid function call: {command}")

        func_name, args_str = match.groups()

        func_map = {
            "dist": self.tools_api.dist,
            "closest": self.tools_api.closest,
            "is_left": self.tools_api.is_left,
            "is_right": self.tools_api.is_right,
            "inside": self.tools_api.inside,
            "most_right": self.tools_api.most_right,
            "most_left": self.tools_api.most_left,
            "middle": self.tools_api.middle,
            "is_empty": self.tools_api.is_empty
        }

        if func_name not in func_map:
            raise ValueError(f"Unknown function: {func_name}")

        parsed_args = self._parse_arguments(args_str)
        result = func_map[func_name](*parsed_args)
        return result

    def _parse_arguments(self, args_str):
        """
        Parses and resolves function arguments from a string.
        Supports nested lists and variable references like <buffer_1>.
        """

        def resolve(node):
            if isinstance(node, ast.Name):
                key = node.id
                if key.startswith('<') and key.endswith('>'):
                    key = key[1:-1]
                if key not in self.masks:
                    raise ValueError(f"Mask '{key}' not found in the provided masks.")
                return self.masks[key]
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.List):
                return [resolve(elt) for elt in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(resolve(elt) for elt in node.elts)
            elif isinstance(node, ast.Str):
                return node.s
            else:
                raise ValueError(f"Unsupported AST node: {ast.dump(node)}")

        # Parse the arguments string as a function call
        tree = ast.parse(f"func({args_str})", mode='eval')
        func_call = tree.body  # ast.Call

        resolved_args = [resolve(arg) for arg in func_call.args]
        return tuple(resolved_args)

if __name__ == "__main__":
    args = parse_args()
    PROJECT_ID = args.project_id
    LOCATION = args.location
    USE_THINK_MODE = args.think_mode
    output_path = args.output_path

    error_budget = 1

    prev_results_path = output_path
    results = []
    convs = []
    answered_ids = set()

    print('Saving results to:', output_path)

    if prev_results_path and os.path.exists(prev_results_path):
        with open(prev_results_path, 'r') as f:
            prev_results = json.load(f)
            results = [item for item in prev_results if item['normalized_answer'] != "-1"]
        with open(prev_results_path.replace('test', 'test_convs'), 'r') as f:
            prev_convs = json.load(f)
            convs = [item for item in prev_convs if item['normalized_answer'] != "-1"]
        answered_ids = {item['id'] for item in results}
        print(f"Loaded {len(results)} previous results.")

    tools = tools_api(dist_model_cfg={'model_path': '../distance_est/ckpt/epoch_5_iter_6831.pth'}, 
                      inside_model_cfg={'model_path': '../inside_pred/ckpt/epoch_4.pth'},
                      small_dist_model_cfg={'model_path': '../distance_est/ckpt/3m_epoch6.pth'},
                      resize=(360, 640),
                      mask_IoU_thres=0.3, inside_thres=0.5,
                      cascade_dist_thres=300, clamp_distance_thres=25)

    with open('../data/test/rephrased_test.json', 'r') as f:
        data = json.load(f)
    
    for idx, item in tqdm(enumerate(data), total=len(data)):
        id = item['id']
        if id in answered_ids:
            continue
            
        agent = Agent(PROJECT_ID, LOCATION, tools, item, think_mode=USE_THINK_MODE)
        agent.set_masks()
        
        attempt = 0
        while attempt < error_budget:
            try:
                answer = agent.set_question()
                answer = agent.format_answer()

                if isinstance(answer, str) and answer.lower() in ['yes', 'no', 'true', 'false']:
                    print(f"Invalid answer format: {answer}.")
                    answer = "-1"
                
                results.append({
                    'id': id,
                    'normalized_answer': str(answer)
                })
                convs.append({
                    'id': id,
                    'normalized_answer': str(answer),
                    'conversation': agent.messages
                })

                break
            except Exception as e:
                attempt += 1
                print(f"Error processing item {id} (attempt {attempt}/{error_budget}): {e}")
                if attempt == error_budget:
                    results.append({
                        'id': id,
                        'normalized_answer': "-1"
                    })
                    convs.append({
                    'id': id,
                    'normalized_answer': "-1",
                    'conversation': agent.messages
                    })
    
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        with open(output_path.replace('test', 'test_convs'), 'w') as f:
            json.dump(convs, f, indent=4)