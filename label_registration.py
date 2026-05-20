import flir_image_extractor
import utils

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pathlib

method = "sift"
plant = "citrus"
num_images = 10
show_registration = False # set to False to skip visualization and just save the warped labels

max_matches = np.inf

thermal_dir = pathlib.Path("data\\" + plant + "\\train_thermal")
visual_dir = pathlib.Path("data\\" + plant + "\\train")
label_dir = pathlib.Path("data\\" + plant + "\\train_label")

images = utils.get_images(label_dir)

flir = flir_image_extractor.FlirImageExtractor()

img_count = 0

for img in images:
    if img_count >= num_images:
        break
    img_count += 1

    image_stem = img.stem
    thermal_path = utils.get_matching_thermal(thermal_dir, image_stem)
    visual_path = utils.get_matching_visual(visual_dir, image_stem)

    flir.process_image(thermal_path)
    thermal_img = flir.get_thermal_np()
    thermal_height, thermal_width = thermal_img.shape[:2]
    embedded_visual = flir.get_rgb_np()

    visual_img = cv2.imread(str(visual_path), cv2.IMREAD_COLOR)
    label_img = cv2.imread(str(img), cv2.IMREAD_COLOR)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)    
    # visual_img = cv2.resize(visual_img, (thermal_width, thermal_height), interpolation=cv2.INTER_AREA)
  
    # print(f"thermal: shape: {thermal_img.shape}, dtype: {thermal_img.dtype}, max: {thermal_img.max()}, min: {thermal_img.min()}")
    # print(f"visual : shape: {visual_img.shape}, dtype: {visual_img.dtype}, max: {visual_img.max()}, min: {visual_img.min()}")

    # print(f"gray thermal: shape: {gray_thermal.shape}, dtype: {gray_thermal.dtype}, max: {gray_thermal.max()}, min: {gray_thermal.min()}")
    # print(f"gray visual: shape: {gray_visual.shape}, dtype: {gray_visual.dtype}, max: {gray_visual.max()}, min: {gray_visual.min()}")

    source = visual_img
    color_source = utils.to_rgb(visual_img)
    gray_source = utils.to_gray(visual_img)

    target = embedded_visual
    color_target = embedded_visual
    gray_target = utils.to_gray(embedded_visual)

    warp_source = label_img

    k1, d1 = utils.find_keypoints_and_descriptors(gray_source, method)
    k2, d2 = utils.find_keypoints_and_descriptors(gray_target, method)

    matches = utils.match_descriptors(d1, d2, method)

    print(f"Found {len(k1)} keypoints in visual image")
    print(f"Found {len(k2)} keypoints in thermal image")
    print(f"Found {len(matches)} matches")

    if max_matches < len(matches):
        matches = matches[:max_matches]

    H, mask = utils.compute_homography(k1, k2, matches)

    warped = cv2.warpPerspective(warp_source, H, (color_target.shape[1], color_target.shape[0]))
    
    red_bgr = np.array([42, 42, 165], dtype=np.uint8)
    green_bgr = np.array([0, 255, 0], dtype=np.uint8)

    # Mask pixels that are NOT red and NOT green
    mask = ~(
        np.all(warped == red_bgr, axis=-1) |
        np.all(warped == green_bgr, axis=-1)
    )

    # Set all other pixels to red
    warped[mask] = red_bgr
    
    output_path = f"data\\{plant}\\train_label_emb\\{image_stem}.png"

    cv2.imwrite(str(output_path), warped)

    # visualize results
    if show_registration:

        overlay = cv2.addWeighted(color_target, 0.7, utils.to_rgb(warped), 0.3, 0)

        source_overlay = cv2.addWeighted(color_source, 0.7, label_img, 0.3, 0)

        marked_visual = cv2.drawKeypoints(source_overlay, k1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        marked_thermal = cv2.drawKeypoints(color_target, k2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        marked_matches = cv2.drawMatches(
            source_overlay, k1, 
            color_target, k2, 
            matches, 
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

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