import re
import json
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import pycocotools.mask as mask_utils

class Mask:
    def __init__(self, object_class: str, object_id: int, region_id: int, rle: Dict):
        self.object_class = object_class
        self.object_id = object_id
        self.region_id = region_id
        self.rle = rle
        self.loaded = None
        self.rle['counts'] = self.rle['counts'].encode('utf-8')
    
    def mask_name(self) -> str:
        return f"{self.object_class}_{self.object_id}"

    def region_name(self) -> str:
        return f"region_{self.region_id}"
    
    def decode_mask(self):
        return mask_utils.decode(self.rle).astype(np.float32)

    def __repr__(self):
        return (f"Mask(object_class='{self.object_class}', "
                f"object_id={self.object_id}, "
                f"region_id={self.region_id}) ")

def parse_masks_from_conversation(conversation: str, rle_data: List[Dict]) -> Dict[str, Mask]:
    """
    Parses mask references from the conversation and builds Mask objects,
    storing them in a dictionary with keys like 'pallet_mask_0'.
    """
    # Regex to find masks: matches <pallet_mask_0>, <transporter_mask_1> etc.
    mask_pattern = re.compile(r"<([a-zA-Z]+)_(\d+)>")

    # Find all matches in the conversation
    matches = mask_pattern.findall(conversation)

    # Dictionary to store masks
    mask_store: Dict[str, Mask] = {}

    # Track object ID counters per object class
    object_counters: Dict[str, int] = {}

    for region_id, match in enumerate(matches):
        object_class = match[0]

        # Assign object ID (increment per class)
        object_id = object_counters.get(object_class, 0)
        object_counters[object_class] = object_id + 1

        # Create Mask instance
        mask_obj = Mask(object_class, object_id, region_id, rle_data[region_id])

        # Build key like 'pallet_mask_0'
        mask_key = f"{object_class}_{object_id}"

        # Store in dictionary
        mask_store[mask_key] = mask_obj

    return mask_store

if __name__ == "__main__":
    
    with open('../data/val/rephrased_val.json', 'r') as f:
        data = json.load(f)
    
    for item in tqdm(data[:100]):

        conversation = item['rephrase_conversations'][0]['value']
        normalized_answer = item['normalized_answer']
        rle_data = item['rle']

        mask_store = parse_masks_from_conversation(conversation, rle_data)

        if len(mask_store) != len(rle_data):
            print(f"Warning: Mismatch in mask count for item {item['id']}. "
                  f"Found {len(mask_store)} masks but expected {len(rle_data)}.")
            print(f"Conversations: {item['conversations'][0]['value']}")
            print(f"rephrase_conversations: {item['rephrase_conversations'][0]['value']}")
            import pdb; pdb.set_trace()

        print(conversation, normalized_answer)

        for key, mask in mask_store.items():
            print(f"{key}: {mask}")
        
        import pdb; pdb.set_trace()