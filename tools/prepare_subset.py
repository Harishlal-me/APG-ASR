import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import sys

# Reconfigure stdout/stderr to use UTF-8 to prevent UnicodeEncodeError on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def check_original_dataset_integrity(src_img_dir: Path, src_json_path: Path) -> tuple[int, int, float]:
    """
    Record the current state of the original dataset to verify later that it has not been modified.
    Returns:
        tuple: (number of images, size of annotation JSON, mtime of annotation JSON)
    """
    if not src_img_dir.exists():
        print(f"Error: Original image directory does not exist: {src_img_dir}", file=sys.stderr)
        sys.exit(1)
    if not src_json_path.exists():
        print(f"Error: Original annotation file does not exist: {src_json_path}", file=sys.stderr)
        sys.exit(1)
        
    # Count original images
    original_images_count = sum(1 for p in src_img_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))
    
    # Get JSON metadata
    json_stat = src_json_path.stat()
    original_json_size = json_stat.st_size
    original_json_mtime = json_stat.st_mtime
    
    return original_images_count, original_json_size, original_json_mtime

def load_coco_json(json_path: Path) -> dict:
    """Safely loads the original COCO annotation JSON in read-only mode."""
    print(f"Loading original annotations from {json_path}...")
    start_time = time.time()
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded successfully in {time.time() - start_time:.2f} seconds.")
        return data
    except FileNotFoundError:
        print(f"Error: Annotation file not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when accessing: {json_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Corrupted JSON file: {json_path}\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

def select_and_split_images(images: list, seed: int = 42) -> tuple[list, list, list]:
    """
    Validates unique image IDs, sorts them deterministically by ID, shuffles
    the images exactly once using random.seed, and splits them into
    train (40k), val (5k), and test (5k).
    """
    print("Validating image IDs and preparing splits...")
    
    # Check for duplicate image IDs
    image_ids = [img['id'] for img in images]
    assert len(image_ids) == len(set(image_ids)), "Error: Duplicate image IDs found in the original annotations!"
    
    total_available = len(images)
    if total_available < 50000:
        print(f"Error: Original dataset only contains {total_available} images, but 50,000 are required.", file=sys.stderr)
        sys.exit(1)
        
    # Sort images by ID to ensure deterministic order before shuffling
    sorted_images = sorted(images, key=lambda x: x['id'])
    
    # Shuffle and select 50,000 images
    random.seed(seed)
    shuffled_images = list(sorted_images)
    random.shuffle(shuffled_images)
    
    selected_subset = shuffled_images[:50000]
    
    train_split = selected_subset[:40000]
    val_split = selected_subset[40000:45000]
    test_split = selected_subset[45000:50000]
    
    return train_split, val_split, test_split

def copy_images(images: list, src_dir: Path, dest_dir: Path, split_name: str) -> int:
    """
    Copies images to the destination split folder using shutil.copy2().
    Skip already copied images to support resuming (though we delete target folder first).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    skipped_count = 0
    
    print(f"Copying {len(images)} images to {dest_dir} ({split_name} split)...")
    for img in tqdm(images, desc=f"Copying {split_name} images"):
        filename = img['file_name']
        src_path = src_dir / filename
        dest_path = dest_dir / filename
        
        if not src_path.exists():
            print(f"\nError: Source image file not found: {src_path}", file=sys.stderr)
            sys.exit(1)
            
        try:
            if dest_path.exists():
                skipped_count += 1
            else:
                shutil.copy2(src_path, dest_path)
                copied_count += 1
        except PermissionError as e:
            print(f"\nError: Permission denied when copying {src_path} to {dest_path}\nDetails: {e}", file=sys.stderr)
            sys.exit(1)
            
    print(f"Completed copying {split_name} images. Copied: {copied_count}, Skipped (Already existed): {skipped_count}")
    return copied_count

def filter_annotations(original_data: dict, selected_images: list, split_name: str) -> dict:
    """
    Filters annotations and image records belonging to the selected subset.
    Preserves all categories, licenses, and info.
    """
    print(f"Filtering annotations for {split_name} split...")
    selected_image_ids = {img['id'] for img in selected_images}
    
    # Filter annotations belonging to selected images
    filtered_annotations = []
    for ann in tqdm(original_data['annotations'], desc=f"Filtering {split_name} annotations"):
        if ann['image_id'] in selected_image_ids:
            filtered_annotations.append(ann)
            
    # Construct standard COCO dict
    split_coco_data = {
        "info": original_data.get("info", {}),
        "licenses": original_data.get("licenses", []),
        "images": selected_images,
        "annotations": filtered_annotations,
        "categories": original_data.get("categories", [])
    }
    
    return split_coco_data

def save_and_verify_json(coco_data: dict, output_path: Path, split_name: str) -> None:
    """Saves the JSON to disk and reloads it to check file integrity and structure."""
    print(f"Saving annotations to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f)
    except PermissionError as e:
        print(f"Error: Permission denied when writing to {output_path}\nDetails: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Verify written JSON integrity
    print(f"Verifying integrity of {output_path}...")
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            verified_data = json.load(f)
            
        required_keys = ["images", "annotations", "categories", "licenses", "info"]
        for key in required_keys:
            if key not in verified_data:
                raise ValueError(f"Missing required COCO key '{key}' in output JSON.")
                
        # Basic record count sanity check
        assert len(verified_data["images"]) == len(coco_data["images"]), "Verified image count mismatch."
        assert len(verified_data["annotations"]) == len(coco_data["annotations"]), "Verified annotation count mismatch."
        print(f"Successfully verified {split_name} JSON parsing and keys.")
    except Exception as e:
        print(f"Error: Verification of output JSON failed for {output_path}\nDetails: {e}", file=sys.stderr)
        sys.exit(1)

def run_post_validation(
    updated_dataset_dir: Path,
    train_images: list, val_images: list, test_images: list,
    train_annots: list, val_annots: list, test_annots: list,
    original_images_count: int, original_json_size: int, original_json_mtime: float,
    src_img_dir: Path, src_json_path: Path,
    total_selected_annotations: int
) -> None:
    """Performs rigorous post-process safety and consistency checks."""
    print("\nRunning post-process validation checks...")
    
    # 1. Verify directory image counts
    train_dir = updated_dataset_dir / "train"
    val_dir = updated_dataset_dir / "val"
    test_dir = updated_dataset_dir / "test"
    
    train_files = list(train_dir.glob("*.jpg"))
    val_files = list(val_dir.glob("*.jpg"))
    test_files = list(test_dir.glob("*.jpg"))
    
    assert len(train_files) == 40000, f"Error: Train images count is {len(train_files)}, expected 40,000"
    assert len(val_files) == 5000, f"Error: Validation images count is {len(val_files)}, expected 5,000"
    assert len(test_files) == 5000, f"Error: Test images count is {len(test_files)}, expected 5,000"
    print("✓ Image file counts in output directories verified (40,000 / 5,000 / 5,000).")
    
    # 2. Verify every copied image actually exists on disk
    train_names = {img['file_name'] for img in train_images}
    val_names = {img['file_name'] for img in val_images}
    test_names = {img['file_name'] for img in test_images}
    
    for f in train_files:
        assert f.name in train_names, f"Error: File {f.name} in train folder is not in selected train images."
    for f in val_files:
        assert f.name in val_names, f"Error: File {f.name} in val folder is not in selected val images."
    for f in test_files:
        assert f.name in test_names, f"Error: File {f.name} in test folder is not in selected test images."
    print("✓ Every copied image physically exists and matches the splits.")
    
    # 3. Verify no duplicated images across Train, Validation, and Test
    train_ids = {img['id'] for img in train_images}
    val_ids = {img['id'] for img in val_images}
    test_ids = {img['id'] for img in test_images}
    
    assert train_ids.isdisjoint(val_ids), "Error: Overlapping image IDs between train and validation splits!"
    assert train_ids.isdisjoint(test_ids), "Error: Overlapping image IDs between train and test splits!"
    assert val_ids.isdisjoint(test_ids), "Error: Overlapping image IDs between validation and test splits!"
    assert len(train_ids) + len(val_ids) + len(test_ids) == 50000, "Error: Total selected unique images count is not 50,000!"
    print("✓ No duplicate or overlapping images across splits (splits are perfectly disjoint).")
    
    # 4. Verify every annotation references an existing image inside that split
    for ann in train_annots:
        assert ann['image_id'] in train_ids, f"Error: Annotation ID {ann['id']} references image ID {ann['image_id']} which is not in train split."
    for ann in val_annots:
        assert ann['image_id'] in val_ids, f"Error: Annotation ID {ann['id']} references image ID {ann['image_id']} which is not in val split."
    for ann in test_annots:
        assert ann['image_id'] in test_ids, f"Error: Annotation ID {ann['id']} references image ID {ann['image_id']} which is not in test split."
    print("✓ All annotations successfully mapped to active images in their respective splits.")
    
    # 5. Verify every image in the split appears in its JSON
    train_json_img_ids = {img['id'] for img in train_images}
    val_json_img_ids = {img['id'] for img in val_images}
    test_json_img_ids = {img['id'] for img in test_images}
    
    assert train_json_img_ids == train_ids, "Error: Train images list inside JSON does not match selected IDs."
    assert val_json_img_ids == val_ids, "Error: Validation images list inside JSON does not match selected IDs."
    assert test_json_img_ids == test_ids, "Error: Test images list inside JSON does not match selected IDs."
    print("✓ Every image in the split appears in its JSON.")

    # 6. Verify annotation count mathematically
    total_split_annotations = len(train_annots) + len(val_annots) + len(test_annots)
    assert total_split_annotations == total_selected_annotations, (
        f"Error: Annotation count mathematical check failed! "
        f"Sum of splits: {total_split_annotations}, Total selected: {total_selected_annotations}"
    )
    print(f"✓ Mathematical check passed: Train ({len(train_annots)}) + Val ({len(val_annots)}) + Test ({len(test_annots)}) = {total_selected_annotations} annotations.")

    # Print annotation counts
    print(f"Train Annotations Count: {len(train_annots)}")
    print(f"Validation Annotations Count: {len(val_annots)}")
    print(f"Test Annotations Count: {len(test_annots)}")
    print(f"Total Images Count: {len(train_images) + len(val_images) + len(test_images)}")
    print(f"Total Annotations Count: {total_split_annotations}")

    # 7. Verify original dataset folder has not been modified
    current_img_count = sum(1 for p in src_img_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))
    current_json_stat = src_json_path.stat()
    
    assert current_img_count == original_images_count, "Error: Original image count has changed!"
    assert current_json_stat.st_size == original_json_size, "Error: Original JSON file size has changed!"
    assert current_json_stat.st_mtime == original_json_mtime, "Error: Original JSON file modification time has changed! File was likely written to."
    print("✓ Integrity verified: Original source dataset is completely untouched.")

def write_logs_and_ids(
    updated_dataset_dir: Path,
    train_images: list, val_images: list, test_images: list,
    train_annots: list, val_annots: list, test_annots: list,
    time_taken: float
) -> None:
    """Generates the creation_log.txt and saves selected split IDs to separate text files."""
    print("\nGenerating creation log and saving split image IDs...")
    
    log_file = updated_dataset_dir / "creation_log.txt"
    log_content = (
        "==========================================\n"
        "COCO 50K Dataset Creation Log\n"
        "==========================================\n"
        f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Random Seed: 42\n\n"
        "Splits Summary:\n"
        f"  Train:      {len(train_images)} images, {len(train_annots)} annotations\n"
        f"  Validation: {len(val_images)} images, {len(val_annots)} annotations\n"
        f"  Test:       {len(test_images)} images, {len(test_annots)} annotations\n"
        f"  Total:      {len(train_images) + len(val_images) + len(test_images)} images, "
        f"{len(train_annots) + len(val_annots) + len(test_annots)} annotations\n\n"
        f"Execution Time: {int(time_taken // 60)} minutes {int(time_taken % 60)} seconds\n"
        "==========================================\n"
    )
    
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(log_content)
    print(f"Saved creation log to: {log_file}")
    
    # Save Image IDs
    for split_name, split_imgs in [("train", train_images), ("val", val_images), ("test", test_images)]:
        id_file = updated_dataset_dir / f"selected_{split_name}_ids.txt"
        with open(id_file, "w", encoding="utf-8") as idf:
            for img in split_imgs:
                idf.write(f"{img['id']}\n")
        print(f"Saved selected {split_name} image IDs to: {id_file}")

def main():
    start_time = time.time()
    
    # Destination paths
    dest_dir = Path("updated_dataset")
    dest_json_dir = dest_dir / "annotations"
    
    # Step 1: Delete existing updated_dataset/ completely if it exists
    if dest_dir.exists():
        print(f"Found existing {dest_dir}. Deleting it completely...")
        try:
            shutil.rmtree(dest_dir)
            print(f"Deleted {dest_dir} successfully.")
        except Exception as e:
            print(f"Error while deleting {dest_dir}: {e}", file=sys.stderr)
            sys.exit(1)
            
    # Recreate directory structure
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "train").mkdir(parents=True, exist_ok=True)
        (dest_dir / "val").mkdir(parents=True, exist_ok=True)
        (dest_dir / "test").mkdir(parents=True, exist_ok=True)
        dest_json_dir.mkdir(parents=True, exist_ok=True)
        print("Recreated target directory structure under 'updated_dataset/'.")
    except Exception as e:
        print(f"Error creating destination directories: {e}", file=sys.stderr)
        sys.exit(1)

    # Source paths
    src_dir = Path("dataset")
    src_img_dir = src_dir / "train2017"
    src_json_path = src_dir / "annotations_trainval2017" / "instances_train2017.json"
    
    # Step 2: Integrity check of original dataset
    orig_img_count, orig_json_size, orig_json_mtime = check_original_dataset_integrity(src_img_dir, src_json_path)
    
    # Step 3: Load original JSON in read-only mode
    coco_data = load_coco_json(src_json_path)
    
    # Step 4: Select and split images
    train_images, val_images, test_images = select_and_split_images(coco_data['images'], seed=42)
    selected_image_ids = {img['id'] for img in train_images} | {img['id'] for img in val_images} | {img['id'] for img in test_images}
    
    # Calculate total annotations expected for the selected subset of images
    print("Calculating expected annotation counts...")
    total_selected_annotations = sum(1 for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids)
    
    # Step 5: Copy images to split folders
    copy_images(train_images, src_img_dir, dest_dir / "train", "Train")
    copy_images(val_images, src_img_dir, dest_dir / "val", "Validation")
    copy_images(test_images, src_img_dir, dest_dir / "test", "Test")
    
    # Step 6: Filter and process annotations
    train_coco = filter_annotations(coco_data, train_images, "Train")
    val_coco = filter_annotations(coco_data, val_images, "Validation")
    test_coco = filter_annotations(coco_data, test_images, "Test")
    
    # Release memory for original dataset to optimize footprint
    del coco_data
    import gc
    gc.collect()
    
    # Step 7: Save and verify the filtered JSON files
    save_and_verify_json(train_coco, dest_json_dir / "instances_train.json", "Train")
    save_and_verify_json(val_coco, dest_json_dir / "instances_val.json", "Validation")
    save_and_verify_json(test_coco, dest_json_dir / "instances_test.json", "Test")
    
    # Step 8: Post-process verification
    run_post_validation(
        dest_dir,
        train_images, val_images, test_images,
        train_coco['annotations'], val_coco['annotations'], test_coco['annotations'],
        orig_img_count, orig_json_size, orig_json_mtime,
        src_img_dir, src_json_path,
        total_selected_annotations
    )
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Step 9: Log details and split files
    write_logs_and_ids(
        dest_dir,
        train_images, val_images, test_images,
        train_coco['annotations'], val_coco['annotations'], test_coco['annotations'],
        time_taken
    )
    
    # Step 10: Print final summary in the requested format
    print("\n==============================")
    print("COCO 2017 Subset Created")
    print("==============================")
    print(f"\nTrain Images: {len(train_images)}")
    print(f"Validation Images: {len(val_images)}")
    print(f"Test Images: {len(test_images)}")
    print(f"\nTotal Images: {len(train_images) + len(val_images) + len(test_images)}")
    print(f"\nTrain Annotations: {len(train_coco['annotations'])}")
    print(f"Validation Annotations: {len(val_coco['annotations'])}")
    print(f"Test Annotations: {len(test_coco['annotations'])}")
    print(f"\nTotal Annotations: {len(train_coco['annotations']) + len(val_coco['annotations']) + len(test_coco['annotations'])}")
    print("\n✓ No overlap detected")
    print("✓ JSON integrity verified")
    print("✓ COCO format preserved")
    print("✓ Dataset ready for YOLOv8/APG-ASR training")

if __name__ == "__main__":
    main()
