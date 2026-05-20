import kornia as K
import kornia.geometry as KG

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import utils
# import pathlib

plant = "citrus"
show_registration = True # set to False to skip visualization and just save the warped labels

def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor)

source = load_timg(f"data\\embedded_visual.png")
target = load_timg(f"data\\thermal.png")

print(f"source shape: {source.shape}, dtype: {source.dtype}, max: {source.max()}, min: {source.min()}")
print(f"target shape: {target.shape}, dtype: {target.dtype}, max: {target.max()}, min: {target.min()}")

registrator = KG.ImageRegistrator("similarity")

# model, intermediate = registrator.register(source, target, output_intermediate_models=True)

H = registrator.register(source, target)
H = H.detach().cpu().numpy()

# If batched, remove batch dimension
if H.ndim == 3:
    H = H[0]

H = H.astype(np.float64)

print(f"Estimated homography:\n{H}")

source_np = source.numpy().squeeze().transpose(1, 2, 0)
target_np = target.numpy().squeeze().transpose(1, 2, 0)

source_np = utils.normalize_to_uint8(source_np)
target_np = utils.normalize_to_uint8(target_np)

warp_source = source_np

warped = cv2.warpPerspective(warp_source, H, (target_np.shape[1], target_np.shape[0]))

overlay = cv2.addWeighted(target_np, 0.75, warped, 0.25, 0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].imshow(source_np)
axes[0, 0].set_title('Source')

axes[0, 1].imshow(target_np)
axes[0, 1].set_title('Target')

axes[1, 0].imshow(warped)
axes[1, 0].set_title('Warped Source')

axes[1, 1].imshow(overlay)
axes[1, 1].set_title('Warped Source Overlay')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()