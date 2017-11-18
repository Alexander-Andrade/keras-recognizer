# from keras.models import load_model
# from keras import backend as K

import samples_utils
import numpy as np
import cv2 as cv
from PIL import Image, ImageMath

test_path = 'train_google_earth/test'

# model = load_model('trained_google_earth.h5')
#
# input_shape = model.input.shape
#
# if K.image_data_format() == 'channels_first':
#     _ , img_width, img_height, _ = input_shape
# else:
#     _ ,img_width, img_height, _ = input_shape


samples_names_gen = samples_utils.file_names(test_path)
file_name = next(samples_names_gen)
im = cv.imread(test_path + '/' + file_name)
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
print("cv format: {}".format(im.shape))
# im_to_model = cv.resize(im, (img_width, img_height)).astype(np.float32)
# im_to_model /= 255

p_im = Image.open(test_path + '/' + file_name)
print("pillow format: {}, {}, {}".format(p_im.format, p_im.size, p_im.mode))

edges = cv.Canny(im, 100, 200)
cv.imshow('edges', edges)
cv.imshow('im', im)

im2, contours, hierarchy = cv.findContours(edges, 1, 2)
cv.drawContours(im, contours, -1, (0,255,0), 1)

cv.imshow('im_with_contours', im)

cv.waitKey(0)
cv.destroyAllWindows()
# print(model.predict(im_to_model, 4))