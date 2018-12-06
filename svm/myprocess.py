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
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import time
from sklearn.cluster import MiniBatchKMeans

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
concate fea from train des to list
"""


def concate_f(list_des):
    concate_l = []
    for d in list_des:
        if d is not None:
            for f in d:
                concate_l.append(f)
    return concate_l


"""
create his for des of img
return ndarray 1D len = k
"""


def create_histogram(des, cluter):
    n_clusters = cluter.get_params()['n_clusters']
    his = np.zeros(n_clusters)
    if des is not None:
        for f in des:
            his[cluter.predict(f.reshape(1, -1))[0]] += 1
    return his


"""
create his for list des of img
return list (ndarray 1D len = k)
"""


def create_histograms(list_des, cluter):
    list_his = [create_histogram(des, cluter) for des in list_des]
    return list_his


"""load model training from disk"""


def load_model_file(path):
    if path.is_file():
        loaded_model = joblib.load(path)
        return loaded_model
    else:
        raise ValueError('can not get training model file')


"""save model training from disk"""


def save_model_file(model, path):
    try:
        joblib.dump(model, path)
    except EOFError as exc:
        raise ValueError('can not save model file', path)


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
        raise ValueError('can not get npy file', path)


"""load ndarray from file"""


def save_npy_file(npy_arr, path):
    try:
        np.save(path, npy_arr)
    except EOFError as exc:
        raise ValueError('can not save npy file', path)


"""metrics for svm"""


def metrics(svm, X_test, Y_test):
    accuracy = svm.score(X_test, Y_test)
    Y_score = svm.predict(X_test)
    precision, recall, fscore, support = precision_recall_fscore_support(
        Y_test, Y_score, average=None)

    # round metrics
    decimal_num = 4
    accuracy = round(accuracy*100, decimal_num)
    precision = [round(pre*100, decimal_num) for pre in precision]
    recall = [round(rec*100, decimal_num) for rec in recall]

    return accuracy, precision, recall


"""train svm model
return svm, elapsed_time"""


def train(X_train, Y_train):
    classifier = svm.LinearSVC()
    # classifier = SVC(kernel='rbf', random_state=0, gamma=.001, C=100)

    t = time.process_time()
    classifier.fit(X_train, Y_train)
    elapsed_time = time.process_time() - t
    elapsed_time = round(elapsed_time, 4)

    return classifier, elapsed_time


"""
clustering features in list des to k cluster
return cluster model, elapsed_time
"""


def clustering(list_des, n_clusters):
    batch_size = n_clusters * 3
    cluster = MiniBatchKMeans(n_clusters=n_clusters,
                              batch_size=batch_size, verbose=1)

    t = time.process_time()
    cluster.fit(list_des)
    elapsed_time = time.process_time() - t
    elapsed_time = round(elapsed_time, 4)

    return cluster, elapsed_time
