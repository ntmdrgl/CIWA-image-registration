import numpy as np
import cv2
import pathlib

def normalize_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

def to_gray(img):
    img = normalize_to_uint8(img)
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_rgb(img):
    img = normalize_to_uint8(img)
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def thermal_to_colormap(img):
    img = normalize_to_uint8(img)
    color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

def find_keypoints_and_descriptors(img, method):
    if method.lower() == "sift":
        feature_extractor = cv2.SIFT_create()
    elif method.lower() == "surf":
        feature_extractor = cv2.xfeatures2d.SURF_create()
    elif method.lower() == "orb":
        feature_extractor = cv2.ORB_create()
    else:
        raise ValueError(f"Unsupported method: {method}")

    keypoints, descriptors = feature_extractor.detectAndCompute(img, None)
    return keypoints, descriptors

def match_descriptors(desc1, desc2, method):
    if method.lower() in ["sift", "surf"]:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method.lower() == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError(f"Unsupported method: {method}")

    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def compute_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    return H, mask


# def crop_black_border(img):
#     """Returns image with black border cropped from an RGB image"""

#     if not (len(img.shape) == 3 and img.shape[2] == 3):
#         raise ValueError("Input image must be a 3-channel RGB image")

#     # convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # threshold the image to create a binary mask of the non-black pixels
#     _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

#     # find bounding box of the non-black pixels
#     coords = cv2.findNonZero(mask)
#     x, y, w, h = cv2.boundingRect(coords)

#     cropped_img = img[y:y+h, x:x+w]

#     return cropped_img

def get_images(img_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return a sorted list of images in a directory"""
    img_suffixes = {".png", ".jpg", ".jpeg"}
    # This is a bunch of python nonsense, but it basically does EXACTLY what the docstring says.
    # Return the sorted list of valid files in the 'img_dir' directory with file extensions in img_suffixes
    return sorted(f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in img_suffixes)

def get_matching_thermal(thermal_dir: pathlib.Path, image_stem: str) -> pathlib.Path | None:
    """
    Find the corresponding thermal image for a given image or label. Returns None if none is found.
    Checks for common image extensions just in case.
    Matches exact stem names or stems followed by a separator (e.g., '_L', '-mask', '.label').
    """
    # Strip '_L' suffix first (labels end in '_L', possibly after '-Visual', but not all of them. Some don't have the _L)
    image_stem = pathlib.Path(image_stem.removesuffix("_L")).stem

    # Then strip '-Visual' suffix and any embedded extension (e.g. 'FLIR_001.JPG-Visual' -> 'FLIR_001')
    image_stem = pathlib.Path(image_stem.removesuffix("-Visual")).stem

    # Keep track of valid extensions and stems for label images
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    valid_prefixes = (f"{image_stem}_", f"{image_stem}-", f"{image_stem}.", f"_{image_stem}")

    # Find candidates and return a valid mask path
    for candidate in thermal_dir.iterdir():
        # Skip anything that isn't an image (specifically of the type we consider, extend valid_exts if necessary)
        if not candidate.is_file() or candidate.suffix.lower() not in valid_exts:
            continue

        # Match valid candidates
        if candidate.stem == image_stem or candidate.stem.startswith(valid_prefixes):
            return candidate

    return None  # Only if we didn't find a match

def get_matching_visual(visual_dir: pathlib.Path, image_stem: str) -> pathlib.Path | None:
    """
    Find the corresponding visual image for a given image or label. Returns None if none is found.
    Checks for common image extensions just in case.
    Matches exact stem names or stems followed by a separator (e.g., '_L', '-mask', '.label').
    """
    # Strip '_L' suffix first (labels end in '_L', possibly after '-Visual', but not all of them. Some don't have the _L)
    image_stem = pathlib.Path(image_stem.removesuffix("_L")).stem

    # Then strip '-Visual' suffix and any embedded extension (e.g. 'FLIR_001.JPG-Visual' -> 'FLIR_001')
    image_stem = pathlib.Path(image_stem.removesuffix("-Visual")).stem

    # Keep track of valid extensions and stems for label images
    valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    valid_prefixes = (f"{image_stem}_", f"{image_stem}-", f"{image_stem}.", f"_{image_stem}", f"{image_stem}", f"{image_stem}-Visual")

    # Find candidates and return a valid mask path
    for candidate in visual_dir.iterdir():
        # Skip anything that isn't an image (specifically of the type we consider, extend valid_exts if necessary)
        if not candidate.is_file() or candidate.suffix.lower() not in valid_exts:
            continue

        # Match valid candidates
        if candidate.stem == image_stem or candidate.stem.startswith(valid_prefixes):
            return candidate

    return None  # Only if we didn't find a match