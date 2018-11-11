import cv2
import numpy as np
from sklearn.externals import joblib
from svm.models import Label, Dataset, Image, Picture
from pathlib import Path
from django.conf import settings
from . import models as md
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import time

"""
Given a path to an image file (PosixPath), returns a cv2 array str -> np.ndarray
"""


def read_image(path):
    return cv2.imread(path)


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
create his for des of img
return ndarray 1D len = k
"""


def create_histogram(des, kmeans):
    k = 25000
    his = np.zeros(k)
    if des is not None:
        for f in des:
            his[kmeans.predict(f.reshape(1, -1))[0]] += 1
    return his


"""load model training from disk"""


def load_model_file(path):
    if path.is_file():
        loaded_model = joblib.load(path)
        return loaded_model
    else:
        raise ValueError('can not get training model file')


"""scaling his"""


def scaling(histograms):
    stdSlr = StandardScaler().fit(histograms)
    return stdSlr.transform(histograms)


"""load ndarray from file"""


def load_npy_file(path):
    if path.is_file():
        ndarray = np.load(path)
        return ndarray
    else:
        raise ValueError('can not get npy file')


"""metrics for svm"""


def metrics(svm, X_test, Y_test):
    accuracy = svm.score(X_test, Y_test)
    Y_score = svm.predict(X_test)
    precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, Y_score, average=None)

    # round metrics
    decimal_num = 4
    accuracy = round(accuracy, decimal_num)
    precision = [round(pre, decimal_num) for pre in precision]
    recall = [round(rec, decimal_num) for rec in recall]

    return accuracy, precision, recall


"""train svm model
return svm, elapsed_time"""


def train(X_train, Y_train):
    classifier = OneVsRestClassifier(svm.LinearSVC())
    classifier.fit(X_train, Y_train)

    return classifier
