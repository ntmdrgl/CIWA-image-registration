import flir_image_extractor
import utils

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib

method = "sift"
plant = "pistachio"
num_images = 1

thermal_dir = pathlib.Path("data\\" + plant + "\\train_thermal")
visual_dir = pathlib.Path("data\\" + plant + "\\train")
label_dir = pathlib.Path("data\\" + plant + "\\train_label")

images = utils.get_images(label_dir)
images = images[:num_images]

flir = flir_image_extractor.FlirImageExtractor()

for img in images:
    image_stem = img.stem
    thermal_path = utils.get_matching_thermal(thermal_dir, image_stem)
    visual_path = utils.get_matching_visual(visual_dir, image_stem)

    flir.process_image(thermal_path)
    thermal_img = flir.get_thermal_np()
    thermal_height, thermal_width = thermal_img.shape[:2]

    visual_img = cv2.imread(str(visual_path), cv2.IMREAD_COLOR)
    visual_img = cv2.resize(visual_img, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA)
  
    print(f"thermal: shape: {thermal_img.shape}, dtype: {thermal_img.dtype}, max: {thermal_img.max()}, min: {thermal_img.min()}")
    print(f"visual : shape: {visual_img.shape}, dtype: {visual_img.dtype}, max: {visual_img.max()}, min: {visual_img.min()}")

    gray_thermal = utils.to_gray(thermal_img)
    gray_visual = utils.to_gray(visual_img)

    # print(f"gray thermal: shape: {gray_thermal.shape}, dtype: {gray_thermal.dtype}, max: {gray_thermal.max()}, min: {gray_thermal.min()}")
    # print(f"gray visual: shape: {gray_visual.shape}, dtype: {gray_visual.dtype}, max: {gray_visual.max()}, min: {gray_visual.min()}")

    k1, d1 = utils.find_keypoints_and_descriptors(gray_visual, method)
    k2, d2 = utils.find_keypoints_and_descriptors(gray_thermal, method)

    matches = utils.match_descriptors(d1, d2, method)

    print(f"Found {len(k1)} keypoints in visual image")
    print(f"Found {len(k2)} keypoints in thermal image")
    print(f"Found {len(matches)} matches")

    H, mask = utils.compute_homography(k1, k2, matches)

    warped = cv2.warpPerspective(visual_img, H, (thermal_img.shape[1], thermal_img.shape[0]))
    warped = utils.to_rgb(warped)
    
    overlay = cv2.addWeighted(utils.thermal_to_colormap(thermal_img), 0.0, warped, 1.0, 0)

    marked_visual = cv2.drawKeypoints(utils.to_rgb(visual_img), k1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    marked_thermal = cv2.drawKeypoints(utils.thermal_to_colormap(thermal_img), k2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    marked_matches = cv2.drawMatches(
        utils.to_rgb(visual_img), k1, 
        utils.thermal_to_colormap(thermal_img), k2, 
        matches, 
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # visualize results

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(marked_visual)
    axes[0, 0].set_title('Source')

    axes[0, 1].imshow(marked_thermal)
    axes[0, 1].set_title('Target')

    axes[1, 0].imshow(marked_matches)
    axes[1, 0].set_title('Feature Matches')

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Warped Source')

    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    raise SystemExit("This is a work in progress. The code is not yet complete.")