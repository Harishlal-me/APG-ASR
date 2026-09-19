import json
from pathlib import Path

# Mapping from COCO 91 categories to YOLO 80 categories (0-indexed)
coco91_to_coco80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11,
    14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21,
    24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
    37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51,
    58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61,
    72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
    82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}

def convert_coco_json(json_path, output_dir):
    print(f"Converting {json_path} to YOLO format in {output_dir}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Create image lookup for fast width/height retrieval
    images_info = {img['id']: (img['width'], img['height'], img['file_name']) for img in data['images']}
    
    # Group annotations by image
    from collections import defaultdict
    annots_by_img = defaultdict(list)
    for ann in data['annotations']:
        annots_by_img[ann['image_id']].append(ann)
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id, img_info in images_info.items():
        w, h, filename = img_info
        txt_filename = Path(filename).with_suffix('.txt').name
        txt_path = output_dir / txt_filename
        
        lines = []
        for ann in annots_by_img.get(img_id, []):
            if 'bbox' not in ann:
                continue
            cat_id = ann['category_id']
            if cat_id not in coco91_to_coco80:
                continue
            yolo_class = coco91_to_coco80[cat_id]
            
            # COCO bbox is [top_left_x, top_left_y, width, height]
            bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
            
            # YOLO expects normalized [x_center, y_center, width, height]
            x_center = (bbox_x + bbox_w / 2) / w
            y_center = (bbox_y + bbox_h / 2) / h
            norm_w = bbox_w / w
            norm_h = bbox_h / h
            
            # Clamp values between 0 and 1 for safety
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
            
    print(f"Finished generating YOLO .txt files for {len(images_info)} images.")

def main():
    base = Path('d:/cdacminor/updated_dataset')
    
    splits = [
        ('instances_train.json', 'train'),
        ('instances_val.json', 'val'),
        ('instances_test.json', 'test')
    ]
    
    for json_name, split_name in splits:
        json_path = base / 'annotations' / json_name
        output_dir = base / split_name
        if json_path.exists():
            convert_coco_json(json_path, output_dir)
        else:
            print(f"[Warning] {json_path} not found.")

if __name__ == '__main__':
    main()
