import pycocotools.mask as mask_utils
from PIL import Image, ImageDraw, ImageFont
import json
import os.path as osp
import numpy as np
import random
import argparse
import os
from tqdm import tqdm

def visualize_masks_and_depth(masks, image_path, depth_path, output_path):
    # Load and process RGB image
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 25)

    text_infos = []

    # Process masks
    for i, mask in enumerate(masks):
        mask = mask_utils.decode(mask)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)  
        colored_mask = Image.new("RGBA", image.size, color)
        overlay.paste(colored_mask, (0, 0), mask_image)

        draw = ImageDraw.Draw(overlay)
        text = f"Region {i}"
        text_size = draw.textbbox((0, 0), text, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        mask_indices = np.argwhere(mask)
        if mask_indices.size > 0:
            min_y, min_x = mask_indices.min(axis=0)
            max_y, max_x = mask_indices.max(axis=0)
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            text_position = (center_x - text_width // 2, center_y - text_height // 2)
            text_infos.append((text, text_position))
    
    draw = ImageDraw.Draw(overlay)
    for text, text_position in text_infos:
        draw.text(text_position, text, fill=(255, 255, 255, 255), font=font)

    # Blend RGB image with mask overlay
    blended_image = Image.alpha_composite(image, overlay)

    # Load and process depth image
    depth = np.array(Image.open(depth_path))
    # Normalize depth to 0-255 range for visualization
    depth_min, depth_max = depth.min(), depth.max()
    depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized).convert('RGB')

    # Create side-by-side visualization
    total_width = blended_image.width + depth_image.width
    combined_image = Image.new('RGB', (total_width, blended_image.height))
    combined_image.paste(blended_image, (0, 0))
    combined_image.paste(depth_image, (blended_image.width, 0))

    # Save the combined visualization
    combined_image.save(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--depth_folder', type=str)  # New argument for depth images
    parser.add_argument('--annotations_file', type=str)
    parser.add_argument('--output_dir', type=str, default='visualization')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()

    image_folder = args.image_folder
    depth_folder = args.depth_folder
    annotations_file = args.annotations_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    base_name = osp.basename(annotations_file)

    annotations = [ann for ann in annotations if ann['image'] == '000711.png']  # Filter for a specific image if needed

    for annotation in tqdm(annotations[:args.num_samples]):
        output_path = osp.join(output_dir, f"{base_name}_{annotation['image']}")
        image_path = osp.join(image_folder, f"{annotation['image']}")
        depth_path = osp.join(depth_folder, f"{annotation['image'].replace('.png', '_depth.png')}")  # Assuming depth images have same names
        masks = annotation['rle']
        visualize_masks_and_depth(masks, image_path, depth_path, output_path)
