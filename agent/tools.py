import os
import sys
from mask import Mask
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from typing import List
import json
from mask import Mask, parse_masks_from_conversation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from distance_est.model import build_dist_model
from inside_pred.model import build_inside_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class tools_api:
    def __init__(self, dist_model_cfg, inside_model_cfg, small_dist_model_cfg, resize=(360,640), mask_IoU_thres=0.3, inside_thres=0.5, cascade_dist_thres=300, clamp_distance_thres=25, img_path=None):
        self.model = build_dist_model(dist_model_cfg)
        self.inside_model = build_inside_model(inside_model_cfg)
        self.small_dist_model = build_dist_model(small_dist_model_cfg)
        self.resize = resize
        self.mask_IoU_thres = mask_IoU_thres
        self.inside_thres = inside_thres
        self.cascade_dist_thres = cascade_dist_thres
        self.clamp_distance_thres = clamp_distance_thres
        self.img_path = img_path
        self.masks = None
    
    def update_masks(self, masks: List[Mask]):
        self.masks = masks

    def update_image(self, img_path):
        self.img_path = img_path
        assert os.path.exists(self.img_path), f"Image path {self.img_path} does not exist."

    def dist(self, mask_1: Mask, mask_2: Mask) -> float:

        if mask_1.object_class.lower() == 'buffer' and self.inside(mask_1, [mask_2]):
            return 0.0
        
        if mask_2.object_class.lower() == 'buffer' and self.inside(mask_2, [mask_1]):
            return 0.0

        rgb = Image.open(self.img_path).convert('RGB')
        rgb = F.resize(rgb, self.resize)
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Decode and resize masks
        mask1_array = mask_1.decode_mask()
        mask2_array = mask_2.decode_mask()
        mask1_img = Image.fromarray(mask1_array)
        mask2_img = Image.fromarray(mask2_array)
        mask1_img = F.resize(mask1_img, self.resize, interpolation=Image.NEAREST)
        mask2_img = F.resize(mask2_img, self.resize, interpolation=Image.NEAREST)
        mask1_resized = np.array(mask1_img).astype(np.float32)
        mask2_resized = np.array(mask2_img).astype(np.float32)

        # Stack inputs
        components = [rgb, mask1_resized[..., None], mask2_resized[..., None]]
        input_tensor = np.concatenate(components, axis=-1)  # H x W x C
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # 1 x C x H x W

        with torch.no_grad():
            predicted_distance = self.model(input_tensor).item()

        if predicted_distance < self.cascade_dist_thres:
            with torch.no_grad():
                predicted_distance = self.small_dist_model(input_tensor).item()
            if predicted_distance < self.clamp_distance_thres:
                predicted_distance = 0.0

        predicted_distance = predicted_distance / 100.0

        return round(predicted_distance, 2)


    def _centroid(self, mask_array: np.ndarray):
        """Compute the centroid (x, y) of a mask."""
        y_indices, x_indices = np.where(mask_array > 0)
        if len(x_indices) == 0:
            return (0, 0)
        return (np.mean(x_indices), np.mean(y_indices))

    def closest(self, mask_A: Mask, masks: List[Mask]) -> str:
        rgb = Image.open(self.img_path).convert('RGB')
        rgb = F.resize(rgb, self.resize)
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Decode and resize mask_A
        maskA_array = mask_A.decode_mask()
        maskA_img = Image.fromarray(maskA_array)
        maskA_img = F.resize(maskA_img, self.resize, interpolation=Image.NEAREST)
        maskA_resized = np.array(maskA_img).astype(np.float32)

        # Prepare the batch
        batch_tensors = []
        for m in masks:
            maskB_array = m.decode_mask()
            maskB_img = Image.fromarray(maskB_array)
            maskB_img = F.resize(maskB_img, self.resize, interpolation=Image.NEAREST)
            maskB_resized = np.array(maskB_img).astype(np.float32)

            # Stack inputs
            components = [rgb, maskA_resized[..., None], maskB_resized[..., None]]
            input_tensor = np.concatenate(components, axis=-1)  # H x W x C
            input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)  # C x H x W
            batch_tensors.append(input_tensor)

        batch_tensor = torch.stack(batch_tensors).to(DEVICE)  # N x C x H x W

        # Model inference
        with torch.no_grad():
            predicted_distances = self.model(batch_tensor).cpu().numpy()

        # Find the mask with the smallest predicted distance
        predicted_distances = predicted_distances / 100.0  # scale back to meters if needed
        min_index = np.argmin(predicted_distances)
        closest_mask = masks[min_index]

        return closest_mask.mask_name()


    def is_left(self, mask_A: Mask, mask_B: Mask) -> bool:
        centroid_A = self._centroid(mask_A.decode_mask())
        centroid_B = self._centroid(mask_B.decode_mask())
        return centroid_A[0] < centroid_B[0]

    def is_right(self, mask_A: Mask, mask_B: Mask) -> bool:
        return not self.is_left(mask_A, mask_B)
    
    def mask_IoU(self, mask_A: Mask, mask_B: Mask) -> float:
        maskA_array = mask_A.decode_mask()
        maskB_array = mask_B.decode_mask()
        intersection = np.logical_and(maskA_array, maskB_array).sum()
        union = np.logical_or(maskA_array, maskB_array).sum()
        iou = intersection / union if union > 0 else 0.0
        return iou

    def inside(self, mask_A: Mask, masks: List[Mask]) -> int:
        rgb = Image.open(self.img_path).convert('RGB')
        rgb = F.resize(rgb, self.resize)
        rgb = np.array(rgb).astype(np.float32) / 255.0

        # Decode and resize mask_A
        maskA_array = mask_A.decode_mask()
        maskA_img = Image.fromarray(maskA_array)
        maskA_img = F.resize(maskA_img, self.resize, interpolation=Image.NEAREST)
        maskA_resized = np.array(maskA_img).astype(np.float32)

        batch_tensors = []

        for m in masks:
            maskB_array = m.decode_mask()
            maskB_img = Image.fromarray(maskB_array)
            maskB_img = F.resize(maskB_img, self.resize, interpolation=Image.NEAREST)
            maskB_resized = np.array(maskB_img).astype(np.float32)

            # Stack RGB, maskA, maskB as channels (adapt as needed)
            components = [rgb, maskA_resized[..., None], maskB_resized[..., None]]
            input_tensor = np.concatenate(components, axis=-1)
            input_tensor = torch.tensor(input_tensor).permute(2, 0, 1)
            batch_tensors.append(input_tensor)

        batch_tensor = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad():
            # Output: logits, convert to 0/1 using torch.round on sigmoid
            logits = self.inside_model(batch_tensor)
            preds = torch.sigmoid(logits)
            preds = torch.round(preds).long().cpu().numpy()  # 1: inside, 0: outside

        count = int(preds.sum())
        return count


    def most_right(self, masks: List[Mask]) -> int:
        max_x = -1
        rightmost_id = -1
        rightmost_mask = None
        for m in masks:
            centroid = self._centroid(m.decode_mask())
            if centroid[0] > max_x:
                max_x = centroid[0]
                rightmost_id = m.object_id
                rightmost_mask = m
        return rightmost_mask.mask_name()

    def most_left(self, masks: List[Mask]) -> int:
        min_x = float('inf')
        leftmost_id = -1
        leftmost_mask = None
        for m in masks:
            centroid = self._centroid(m.decode_mask())
            if centroid[0] < min_x:
                min_x = centroid[0]
                leftmost_id = m.object_id
                leftmost_mask = m
        return leftmost_mask.mask_name()
    
    def middle(self, masks: List[Mask]) -> str:
        # return the mask that is in the middle of all masks
        if not masks:
            raise ValueError("No masks provided to find the middle mask.")
        assert len(masks) == 3, "The middle function requires exactly 3 masks."
        centroids = []
        for m in masks:
            centroid_x, _ = self._centroid(m.decode_mask())
            centroids.append((centroid_x, m))

        # Sort masks by centroid_x
        centroids.sort(key=lambda x: x[0])

        # Return the mask name of the middle one
        middle_mask = centroids[1][1]
        return middle_mask.mask_name()

    def is_empty(self, transporter_masks: List[Mask]) -> str:
        """
        For each transporter mask, calculate the maximum IoU with all pallet masks.
        Return the transporter mask with the smallest maximum IoU.
        """
        min_max_IoU = float('inf')
        selected_transporter = None

        pallet_masks = [m for m in self.masks.values() if 'pallet' in m.object_class.lower()]

        for transporter in transporter_masks:
            if transporter.object_class.lower() != 'transporter':
                continue

            max_IoU = 0.0
            for pallet in pallet_masks:
                iou = self.mask_IoU(transporter, pallet)
                if iou > max_IoU:
                    max_IoU = iou

            if max_IoU < min_max_IoU:
                min_max_IoU = max_IoU
                selected_transporter = transporter

        if selected_transporter is not None:
            return selected_transporter.mask_name()
        else:
            raise ValueError("No transporter masks found in the provided list.")