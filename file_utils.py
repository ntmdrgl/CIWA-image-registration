import pathlib

def get_images(img_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return a sorted list of images in a directory"""
    img_suffixes = {".png", ".jpg", ".jpeg"}
    # This is a bunch of python nonsense, but it basically does EXACTLY what the docstring says.
    # Return the sorted list of valid files in the 'img_dir' directory with file extensions in img_suffixes
    return sorted(f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in img_suffixes)