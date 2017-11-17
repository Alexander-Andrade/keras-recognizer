from os import listdir
from os.path import isfile, join
from PIL import Image, ImageMath


def file_names(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


# im = Image.open('train_google_earth/preprocess\\images\\22828930_15.tiff').convert('RGB')
# map = Image.open('train_google_earth/preprocess\\target_maps\\22828930_15.tif').convert('RGB')
#
# print(im.format, im.size, im.mode)
# print(map.format, map.size, map.mode)
#
# r, g, b = map.split()
# new_map = Image.merge("RGBA", (r, r, r, r))
# new_map.show()
#
# masked = Image.composite(im, new_map, mask=new_map).convert('RGB')
# print(masked.format, masked.size, masked.mode)
# masked.show()