import cv2
import numpy as np
from sklearn.externals import joblib

"""
Given a path to an image file (PosixPath), returns a cv2 array str -> np.ndarray
"""


def read_image(path):
    return cv2.imread(path)

    # if path.is_file():
    #     return cv2.imread(str(path))
    # else:
    #     raise ValueError('Path provided is not a valid file: {}'.format(path))


"""
feature extract from img
return des: nparray
"""


def feature_extract(image_path):
    sift = cv2.xfeatures2d.SIFT_create()
    img = read_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    return des


"""
load the kmeans from disk
"""


def load_kmeans():
    path = 'flower1-kmeans-10000.sav'
    loaded_kmeans = joblib.load(path)
    return loaded_kmeans


"""
create his for des of img
return ndarray 1D len = k
"""


def create_histogram(des, kmeans):
    k = 10000
    his = np.zeros(k)
    if des is not None:
        for f in des:
            his[kmeans.predict(f.reshape(1, -1))[0]] += 1
    return his


"""
load the model from disk
"""


def load_svm():
    path = 'flower1-svm-10000.sav'
    loaded_svm = joblib.load(path)
    return loaded_svm
