import os
import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAMDetector:

    def __init__(self, model_type="vit_b", device="cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"

        sam_checkpoint = "models/sam_vit_b_01ec64.pth"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(self.device)

        mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=8,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=100,
        box_nms_thresh=0.7
        )
        self.mask_generator = mask_generator


    def detect_animals(self, image_path, save_crops=False):
        debug_folder = "debug_crops"
        os.makedirs(debug_folder, exist_ok=True)
        image = cv2.imread(image_path)

        # 🔴 image load check
        if image is None:
            print("❌ Image not found:", image_path)
            return []
        image = cv2.resize(image, (800,600))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        masks = self.mask_generator.generate(image_rgb)
        # sort masks by area (largest first)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)

        # keep only top 5 objects (max animals expected)
        masks = masks[:3]
        print("Total SAM masks:", len(masks))
        crops = []

        for i, mask in enumerate(masks):

            y_indices, x_indices = np.where(mask["segmentation"])

            if len(y_indices) == 0:
                continue

            x1, x2 = x_indices.min(), x_indices.max()
            y1, y2 = y_indices.min(), y_indices.max()
            w = x2 - x1
            h = y2 - y1

            area = w * h
            image_area = image.shape[0] * image.shape[1]

            # ignore tiny segments
            if area < image_area * 0.05:
                continue

            # ignore weird shapes
            aspect_ratio = w / h
            if aspect_ratio < 0.15 or aspect_ratio > 5:
                continue
            if w < 60 or h < 60:
                continue
            cropped = image[y1:y2, x1:x2]
            cropped = cv2.resize(cropped, (224, 224))
            crops.append(cropped)

            if save_crops:
                cv2.imwrite(os.path.join(debug_folder, f"debug_crop_{i}.jpg"), cropped)

        return crops