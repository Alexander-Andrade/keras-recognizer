from keras.models import load_model
from keras import backend as K

import samples_utils
import numpy as np
import cv2 as cv

test_path = 'train_google_earth/test'


def elongation(m):
    sum_part = m['m20'] + m['m02']
    sqrt_part = (((m['m20'] - m['m02']) ** 2) + 4*m['m11']**2) ** 0.5
    return (sum_part + sqrt_part) / (sum_part - sqrt_part)

model = load_model('trained_google_earth.h5')

input_shape = model.input.shape

if K.image_data_format() == 'channels_first':
    _ , img_width, img_height, _ = input_shape
else:
    _ ,img_width, img_height, _ = input_shape


samples_names_gen = samples_utils.file_names(test_path)
next(samples_names_gen)
next(samples_names_gen)
next(samples_names_gen)
file_name = next(samples_names_gen)
im = cv.imread(test_path + '/' + file_name)
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
out_im = im.copy()
print("cv format: {}".format(im.shape))

edges = cv.Canny(im, 100, 200)
# cv.imshow('edges', edges)
# cv.imshow('im', im)

im2, contours, hierarchy = cv.findContours(edges, 1, 2)
contours = list(filter(lambda c: cv.contourArea(c) > 200, contours))

# cv.drawContours(im, contours, -1, (255,0 , 0), 1)
# cv.imshow('orig contours', im)

# contours = list(filter(lambda c: elongation(cv.moments(c)) < 3000, contours))
# contours = list(filter(lambda c: cv.arcLength(c, False) > 100, contours))
# contours = list(map(lambda c: cv.approxPolyDP(c, 2.0, True), contours))
# contours = list(map(lambda c: cv.convexHull(c), contours))

# cv.drawContours(im, contours, -1, (0,255,0), 1)
# cv.imshow('filtered', im)
bounding_rects = list(map(lambda c: cv.boundingRect(c), contours))
for x, y, w, h in bounding_rects:
    croped_im = im[y:y + h, x:x + w]  # Crop from x, y, w, h -> 100, 200, 300, 400
    im_to_model = cv.resize(croped_im, (img_width, img_height))
    cv.imshow('part' + str(i), im_to_model)
    im_to_model = im_to_model.astype(np.float32)
    im_to_model /= 255
    im_to_model = np.expand_dims(im_to_model, axis=0)
    cv.rectangle(out_im, (x, y), (x + w, y + h), (255, 0, 0), 1)
    # print(model.predict_classes(im_to_model))
    if model.predict_classes(im_to_model)[0][0] == 1:
        cv.rectangle(out_im, (x, y), (x+w, y+h), (0, 0, 255), 1)


cv.imshow('rects', out_im)

# im_to_model = np.expand_dims(im_to_model, axis=0)
# print(im_to_model.shape)
# print(model.predict(im_to_model))

cv.waitKey(0)
cv.destroyAllWindows()