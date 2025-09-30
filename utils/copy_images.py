# Copy images to another folder
import shutil
from pathlib import Path

def copy_images(source_dir, dest_dir):
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Move all files from source to destination
    if source_dir.exists():
        for file_path in source_dir.iterdir():
            if file_path.is_file():
                dest_file = dest_dir / file_path.name
                shutil.copy2(str(file_path), str(dest_file))
                print(f"Copied: {file_path.name}")
    else:
        print(f"Source directory {source_dir} does not exist")

if __name__ == "__main__":
    source_dir = Path("data/pokemon_sprites/home")
    dest_dir = Path("data/test")
    copy_images(source_dir, dest_dir)
