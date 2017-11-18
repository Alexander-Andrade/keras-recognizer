import numpy as np
from PIL import Image, ImageMath
import samples_utils

images_path = 'train_google_earth/preprocess/images'
target_maps_path = 'train_google_earth/preprocess/target_maps'
result_path = 'train_google_earth/preprocess/result'

n_cols = 5
n_rows = 5


def tile_traverse_gen(width, height, rows, cols):
    width_step = width // cols
    height_step = height // rows

    top_left = np.array([0, 0])
    bottom_right = np.array([width_step, height_step])

    while bottom_right[1] <= height:
        top_left[0] = 0
        bottom_right[0] = width_step
        while bottom_right[0] <= width:
            yield (top_left[0], top_left[1], bottom_right[0], bottom_right[1])
            top_left[0] += width_step
            bottom_right[0] += width_step
        top_left[1] += height_step
        bottom_right[1] += height_step


def apply_mask(image, mask):
    r, g, b = mask.split()
    corrected_mask = Image.merge("RGBA", (r, r, r, r))
    return Image.composite(image, corrected_mask, mask=corrected_mask).convert('RGB')


for image_name in samples_utils.file_names(images_path):
    im = Image.open(images_path + '/' + image_name).convert('RGB')
    mask_name = samples_utils.change_filename_ext(image_name, 'tif')
    mask = Image.open(target_maps_path + '/' + mask_name).convert('RGB')

    masked_im = apply_mask(im, mask)
    width, height = masked_im.size
    for i, box in enumerate(tile_traverse_gen(width, height, n_rows, n_cols)):
        im_part_name = samples_utils.gen_image_part_name(image_name, i, 'png')
        masked_im.crop(box).save(result_path + '/' + im_part_name, 'PNG')
