"""
Dataset download and extraction utility for Aircraft Damage Dataset.
"""
import tarfile
import urllib.request
import os
import shutil


def download_and_extract(progress_callback=None):
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
    tar_filename = "aircraft_damage_dataset_v1.tar"
    extracted_folder = "aircraft_damage_dataset_v1"
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tar_path = os.path.join(base_dir, tar_filename)
    extract_path = os.path.join(base_dir, extracted_folder)

    # Download
    if progress_callback:
        progress_callback("Downloading dataset...")
    
    def reporthook(count, block_size, total_size):
        pass

    urllib.request.urlretrieve(url, tar_path, reporthook)
    
    if progress_callback:
        progress_callback("Download complete. Extracting...")

    # Remove existing folder
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    # Extract
    with tarfile.open(tar_path, "r") as tar_ref:
        tar_ref.extractall(base_dir)
    
    if progress_callback:
        progress_callback("Extraction complete!")

    return extract_path


def get_dataset_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "aircraft_damage_dataset_v1")


def dataset_exists():
    path = get_dataset_path()
    return (
        os.path.exists(path)
        and os.path.exists(os.path.join(path, "train"))
        and os.path.exists(os.path.join(path, "valid"))
        and os.path.exists(os.path.join(path, "test"))
    )


def count_images(split_dir):
    """Count images per class in a split directory."""
    counts = {}
    if not os.path.exists(split_dir):
        return counts
    for cls in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return counts
