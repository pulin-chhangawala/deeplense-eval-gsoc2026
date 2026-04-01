"""Extract and organize datasets for evaluation tasks."""
import zipfile
from pathlib import Path
import shutil
import os
import subprocess

def download_from_google_drive(file_id, output_path):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call(["pip", "install", "-q", "gdown"])
        import gdown
    
    print(f"Downloading {output_path}...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    return Path(output_path).exists()

def extract_and_organize():
    work_dir = Path(".")
    
    # Test I file ID: 1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ
    test1_zip = work_dir / "test1_dataset.zip"
    # Test V file ID: 1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5
    test5_zip = work_dir / "test5_dataset.zip"
    
    # Download datasets if they don't exist
    if not test1_zip.exists():
        print("\n=== DOWNLOADING TEST I (Multi-Class Classification) ===")
        download_from_google_drive("1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ", str(test1_zip))
    
    if not test5_zip.exists():
        print("\n=== DOWNLOADING TEST V (Lens Finding) ===")
        download_from_google_drive("1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5", str(test5_zip))
    
    # Extract Test I (Multi-class classification)
    print(f"\n=== EXTRACTING TEST I ===")
    print(f"Extracting Test I dataset from {test1_zip}...")
    with zipfile.ZipFile(test1_zip, 'r') as z:
        z.extractall("temp_test1")
    
    # Find and organize task1 data
    task1_dir = (Path("temp_test1") / "dataset").resolve()
    if task1_dir.exists():
        if Path("task1_data").exists():
            shutil.rmtree("task1_data")
        shutil.move(str(task1_dir), "task1_data")
        print(f"✓ Task 1 data organized in task1_data/")
    else:
        print(f"⚠ Could not find dataset directory in temp_test1")
    
    # Extract Test V (Lens Finding)
    print(f"\n=== EXTRACTING TEST V ===")
    print(f"Extracting Test V dataset from {test5_zip}...")
    if Path("task2_data").exists():
        shutil.rmtree("task2_data")
    with zipfile.ZipFile(test5_zip, 'r') as z:
        z.extractall("task2_data")
    print(f"✓ Task 2 data extracted to task2_data/")
    
    # Clean up temp directory
    if Path("temp_test1").exists():
        shutil.rmtree("temp_test1")
    
    # Check structure
    print("\n=== VERIFICATION ===")
    if Path("task1_data").exists():
        subdirs = sorted([d.name for d in Path('task1_data').glob('*') if d.is_dir()])
        print(f"Task 1 structure: {subdirs}")
        for subdir in subdirs:
            count = len(list(Path(f'task1_data/{subdir}').rglob('*.npy')))
            print(f"  └─ {subdir}: {count} files")
    
    if Path("task2_data").exists():
        subdirs = sorted([d.name for d in Path('task2_data').glob('*') if d.is_dir()])
        print(f"Task 2 structure: {subdirs}")
        for subdir in subdirs:
            count = len(list(Path(f'task2_data/{subdir}').rglob('*.npy')))
            print(f"  └─ {subdir}: {count} files")

if __name__ == "__main__":
    extract_and_organize()
