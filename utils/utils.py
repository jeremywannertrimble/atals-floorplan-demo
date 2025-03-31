import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
# import fitz 
import io

def pad_image(img, patch_size=640):
    """Pads the image to ensure it's evenly divisible by the patch size."""
    width, height = img.size
    new_width = ((width + patch_size - 1) // patch_size) * patch_size
    new_height = ((height + patch_size - 1) // patch_size) * patch_size
    if width == new_width and height == new_height:
        return img
    padded_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    padded_img.paste(img, (0, 0))
    return padded_img

def create_patches(img, overlap_ratio, patch_size=640):
    """Splits the image into overlapping patches if needed."""
    patches_dic = {}
    width, height = img.size
    
    if width <= patch_size and height <= patch_size:
        patches_dic["0-0"] = img
        return patches_dic, 0
    
    overlap = int(patch_size * float(overlap_ratio))
    step = patch_size - overlap
    
    for i, x in enumerate(range(0, width, step)):
        for j, y in enumerate(range(0, height, step)):
            # Get actual patch dimensions before padding
            actual_w = min(patch_size, width - x)
            actual_h = min(patch_size, height - y)
            
            patch = img.crop((x, y, x + actual_w, y + actual_h))
            if patch.size[0] < patch_size or patch.size[1] < patch_size:
                # Store original dimensions before padding
                patches_dic[f'{i}-{j}'] = {
                    'image': pad_image(patch, patch_size),
                    'original_size': (actual_w, actual_h)
                }
            else:
                patches_dic[f'{i}-{j}'] = {
                    'image': patch,
                    'original_size': (actual_w, actual_h)
                }
    
    return patches_dic, step

def get_yolo_masks(img, model):
    """Gets masks from YOLO segmentation model."""
    try:
        results = model.predict(img)
        masks = results[0].masks
        if masks is None:
            return None, None
        bboxes = results[0].boxes
        pred_cls = bboxes.cls.tolist()
        return masks.data.numpy(), pred_cls
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

def generate_patches_masks(patches_images_dic, model, progress_bar):
    """Processes YOLO predictions for each patch."""
    dic = {}
    total_patches = len(patches_images_dic)
    track_progress = 0

    for patch_id, patch_data in patches_images_dic.items():
        dic[patch_id] = {}
        pred_masks, pred_cls = get_yolo_masks(patch_data['image'], model)
        track_progress += 1
        progress_bar.progress(track_progress / total_patches, text="Walls Extraction")
        if pred_masks is None:
            continue
        for mask, cls in zip(pred_masks, pred_cls):
            dic[patch_id].setdefault(cls + 1.0, []).append(mask)
    
    return dic

def merging_patches(w, h, patches_masks_dic, step, room_mask=None):
    """Reconstructs the full mask from patches if necessary."""
    if step == 0:
        return next(iter(patches_masks_dic.values())).get(0, np.zeros((h, w), dtype=np.uint8))
    
    # Calculate padded dimensions
    patch_size = 640
    padded_w = ((w + patch_size - 1) // patch_size) * patch_size
    padded_h = ((h + patch_size - 1) // patch_size) * patch_size
    
    # Create mask with padded dimensions
    padded_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)
    
    for patch_id, class_masks in patches_masks_dic.items():
        x_patch_id, y_patch_id = map(int, patch_id.split("-"))
        x_offset = x_patch_id * step
        y_offset = y_patch_id * step
        
        # Ensure we don't exceed padded dimensions
        x_end = min(x_offset + patch_size, padded_w)
        y_end = min(y_offset + patch_size, padded_h)
        patch_w = x_end - x_offset
        patch_h = y_end - y_offset
        
        for cls, masks in class_masks.items():
            if cls == 1:
                continue
            for mask in masks:
                # Resize mask to match the actual patch size we can use
                mask_resized = cv2.resize(mask, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST)
                padded_mask[y_offset:y_end, x_offset:x_end] = np.maximum(
                    padded_mask[y_offset:y_end, x_offset:x_end],
                    mask_resized * cls
                )
    
    # Crop back to original dimensions
    final_mask = padded_mask[:h, :w]
    
    if room_mask is not None:
        room_mask = cv2.resize(room_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        final_mask[room_mask == 1] = 1
    
    return final_mask

def visualize_mask(mask, original_img, output_path="mask_overlay.png"):
    """Overlays segmentation mask on the original image."""
    color_map = {
        1: (200, 255, 150),  # pink / rooms
        2: (255, 255, 0),    # yellow / walls
        3: (0, 255, 0),      # green / door -> pink (255, 0, 255)
        4: (0, 0, 255)       # blue / windows
    }
    
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_mask[mask == label] = color
    
    plt.figure(figsize=(10, 10))
    plt.imshow(original_img)
    plt.imshow(color_mask, alpha=0.5)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return Image.open(output_path) 


#def pdf_to_img(pdf_path):
    """Convert PDF to image."""
    # doc = fitz.open(pdf_path)
    # page = doc.load_page(0)
    # pixmap = page.get_pixmap(dpi=150)
    # img_bytes = pixmap.tobytes()
    # return Image.open(io.BytesIO(img_bytes)) 