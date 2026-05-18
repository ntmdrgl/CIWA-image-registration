import flir_image_extractor
import file_utils

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib

plant = "pistachio"
num_images = 1

thermal_dir = pathlib.Path("data\\" + plant + "\\train_thermal")

images = file_utils.get_images(thermal_dir)
images = images[:num_images]

flir = flir_image_extractor.FlirImageExtractor()

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

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def to_rgb(img):
    img = normalize_to_uint8(img)

    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

def thermal_to_colormap(img):
    img_uint8 = normalize_to_uint8(img)
    color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

def image_registration(visual_image, thermal_image, good_match_percent=0.50):
    rgb_source = to_rgb(visual_image)
    rgb_target = thermal_to_colormap(thermal_image)

    gray_source = to_gray(visual_image)
    gray_target = to_gray(thermal_image)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray_source, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_target, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:int(len(matches) * good_match_percent)]

    print(f'Number of keypoints in source image: {len(keypoints1)}')
    print(f'Number of keypoints in target image: {len(keypoints2)}')
    print(f'Number of matches: {len(matches)}')

    if len(matches) < 4:
        print("Not enough matches found to compute homography.")
        exit()


    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(f'Homography matrix:\n{H}')

    result = cv2.warpPerspective(visual_image, H, (thermal_image.shape[1], thermal_image.shape[0]))
    result_rgb = to_rgb(result)

    match_img = cv2.drawMatches(rgb_source, keypoints1, rgb_target, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    overlay = cv2.addWeighted(rgb_target, 0.0, result_rgb, 1.0, 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(rgb_source)
    axes[0, 0].set_title('Source')

    axes[0, 1].imshow(rgb_target)
    axes[0, 1].set_title('Target')

    axes[1, 0].imshow(match_img)
    axes[1, 0].set_title('Feature Matches')

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Registration Overlay')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def crop_black_border(img, threshold=10):
    """
    Removes black border from RGB image.
    Works for visual_image shape (H, W, 3).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mask = gray > threshold

    coords = np.argwhere(mask)
    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img[y0:y1, x0:x1]


def preprocess_visual_for_thermal(visual_image, thermal_image):
    visual_crop = crop_black_border(visual_image)

    thermal_h, thermal_w = thermal_image.shape[:2]

    visual_resized = cv2.resize(
        visual_crop,
        (thermal_w, thermal_h),
        interpolation=cv2.INTER_AREA
    )

    return visual_resized

for image in images:
    flir.process_image(image)

    thermal_image = flir.get_thermal_np()
    visual_image = flir.get_rgb_np()

    visual_preprocessed = preprocess_visual_for_thermal(
        visual_image,
        thermal_image
    )

    print("thermal:", thermal_image.shape)
    print("visual original:", visual_image.shape)
    print("visual preprocessed:", visual_preprocessed.shape)

    image_registration(visual_preprocessed, thermal_image, 1.0)