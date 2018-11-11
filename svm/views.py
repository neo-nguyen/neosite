from django.shortcuts import render
from pathlib import Path
from svm.models import Label, Dataset, Image, Picture
from django.http import HttpResponse
import os
from django.db.models import Count
from django.db import transaction
from django.db import connection
from django.contrib import messages
from .form import PictureUploadForm
from django.template import RequestContext

from . import myprocess as mp
from . import models as md
import numpy as np
from django.conf import settings
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
# Create your views here.


def index(request):
    return HttpResponse("Welcome my app!")


"""from a valid dataset_path
read names of label -> add to DB"""


def create_labels(dataset_path):
    train_dir = Path(dataset_path) / 'train'
    subdirs = [subdir for subdir in train_dir.iterdir() if subdir.is_dir()]
    if subdirs:
        Label.objects.all().delete()
        for subdir in subdirs:
            label_name = subdir.parts[-1]
            label = Label(label_name=label_name)
            label.save()


"""names of dataset type -> add to DB"""


def create_datasets():
    Dataset.objects.all().delete()
    dts = ['train', 'test', 'validation']
    for dt in dts:
        dataset = Dataset(dataset_name=dt)
        dataset.save()


"""
read path img from folder contain img
return list img paths: PosixPath
"""


def get_file_paths(root, files_of_type):
    file_paths = []
    for cwd, folders, files in os.walk(root):
        for fname in files:
            if os.path.splitext(fname)[1] in files_of_type:
                path = Path(cwd)
                file_paths.append(path / fname)
    return file_paths


"""from a valid dataset_path
read path of images -> add to DB"""


@transaction.non_atomic_requests
def create_images(dataset_path):
    Image.objects.all().delete()

    img_paths = get_file_paths(dataset_path, '.png .jpg .jpeg .JPEG .JPG')
    if img_paths:
        for img_path in img_paths:
            # #get dataset id
            dts_name = img_path.parts[-3]
            dts_id = Dataset.objects.get(dataset_name=dts_name).id

            # #get label id
            lb_name = img_path.parts[-2]
            lb_id = Label.objects.get(label_name=lb_name).id

            image = Image(image_path=img_path,
                          label_id=lb_id, dataset_id=dts_id)
            image.save()

        transaction.commit()


def read_dataset(dataset_path):
    # reset auto increment in sqlite
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM SQLite_sequence")

    # create data from folder
    create_datasets()
    create_labels(dataset_path)
    create_images(dataset_path)


"""get dataset info from DB, make sure create DB before"""


def dataset_info(request):
    if (request.POST.get('dataset_path')):
        dataset_path = request.POST['dataset_path']
        # validate path -> if ok ->
        read_dataset(dataset_path)
        messages.success(request, 'Read dataset successfully!')

    labels, label_nums = Label.objects.all(), Label.objects.count()

    image_nums = Image.objects.count()

    # return list[label and count images by label]
    image_count = Label.objects.annotate(num_images=Count('image'))

    # build a dic {label name: num image}
    image_count_by_label = {}
    for i in range(len(image_count)):
        image_count_by_label[image_count[i].description] = image_count[i].num_images

    context = {
        'labels': labels,
        'label_nums': label_nums,
        'image_nums': image_nums,
        'image_count_by_label': image_count_by_label,
    }

    return render(request, 'svm/dataset_info.html', context)


"""show each class 1 random img"""


def show_picture(request):
    img = Image()
    imgs = img.get_random_per_label()
    return render(request, 'svm/show_picture.html', locals())


"""get name of dataset from path of 1 img"""


def get_name_dataset():
    if Image.objects.count() != 0:
        image = Image.objects.filter().latest('id')
        path = Path(image.image_path)
        return path.parts[-4]
    else:
        raise ValueError('can not get data name, no image in data')


"""build path to upload file"""


def path_upload_file(file_name):
    return Path(settings.MEDIA_ROOT + md.upload_dir) / file_name


"""feature extract from list Image.objects
return list des"""


def feature_extracting(image_objs):
    list_des = []
    if len(image_objs) != 0:
        for image in image_objs:
            des = mp.feature_extract(image.image_path)
            list_des.append(des)

    return list_des


"""feature extract from all img in DB
save file des to upload_dir"""


def extracting(request):
    dataset_name = get_name_dataset()

    if Image.objects.count() != 0:
        train_imgs = Image.objects.filter(dataset_id=1)
        train_des = feature_extracting(train_imgs)
        train_path = path_upload_file(dataset_name + '-train-des.npy')
        mp.save_npy_file(train_des, train_path)

        test_imgs = Image.objects.filter(dataset_id=2)
        test_des = feature_extracting(test_imgs)
        test_path = path_upload_file(dataset_name + '-test-des.npy')
        mp.save_npy_file(test_des, test_path)

        train_des_len = len(train_des)
        test_des_len = len(test_des)

        creat_train_test_labels()
    else:
        raise ValueError('can not extract, no image in data')

    return render(request, 'svm/extracting.html', locals())


"""creating a visual vocabulary by kmeans from train des
save kmeans model to upload_dir"""


def vocabulary(request):
    dataset_name = get_name_dataset()

    train_des = mp.load_npy_file(
        path_upload_file(dataset_name + '-train-des.npy'))
    concate_train_des = mp.concate_f(train_des)

    n_clusters = 100
    cluster, elapsed_time = mp.clustering(concate_train_des, n_clusters)
    cluster_path = path_upload_file(dataset_name + '-kmeans.sav')
    mp.save_model_file(cluster, cluster_path)

    return render(request, 'svm/vocabulary.html', locals())


"""creating his by kmeans from train, test des
save train, test his to upload_dir"""


def histogram(request):
    dataset_name = get_name_dataset()

    train_des = mp.load_npy_file(
        path_upload_file(dataset_name + '-train-des.npy'))
    test_des = mp.load_npy_file(
        path_upload_file(dataset_name + '-test-des.npy'))

    cluster = mp.load_model_file(
        path_upload_file(dataset_name + '-kmeans.sav'))

    train_his = mp.create_histograms(train_des, cluster)
    test_his = mp.create_histograms(test_des, cluster)

    train_his_path = path_upload_file(dataset_name + '-train-his.npy')
    mp.save_npy_file(train_his, train_his_path)
    test_his_path = path_upload_file(dataset_name + '-test-his.npy')
    mp.save_npy_file(test_his, test_his_path)

    train_his_len = len(train_his)
    test_his_len = len(test_his)

    return render(request, 'svm/histogram.html', locals())


"""training process page"""


def training(request):
    dataset_name = get_name_dataset()

    train_his = mp.load_npy_file(path_upload_file(
        dataset_name + '-train-his.npy'))
    train_his = mp.scaling(train_his)

    train_labels = mp.load_npy_file(
        path_upload_file(dataset_name + '-train-labels.npy'))

    classifier, elapsed_time = mp.train(train_his, train_labels)
    mp.save_model_file(classifier, path_upload_file(dataset_name + '-svm.sav'))

    return render(request, 'svm/training.html', locals())


"""evaluating model page"""


def evaluating(request):
    dataset_name = get_name_dataset()

    test_his = mp.load_npy_file(path_upload_file(
        dataset_name + '-test-his.npy'))
    test_his = mp.scaling(test_his)
    test_labels = mp.load_npy_file(
        path_upload_file(dataset_name + '-test-labels.npy'))

    classifier = mp.load_model_file(
        path_upload_file(dataset_name + '-svm.sav'))

    accuracy, precision, recall = mp.metrics(classifier, test_his, test_labels)

    labels = Label.objects.all()

    # build a dic {label description: [precision, recall]}
    label_metrics = {}
    for label in labels:
        metrics = [precision[label.id - 1], recall[label.id - 1]]
        label_metrics[label.description] = metrics

    return render(request, 'svm/evaluating.html', locals())


"""creating list labels from list Image.obj"""


def creat_list_labels(image_objs):
    # tru 1 de giong voi index khi train tren jupyter
    list_labels = [img.label_id - 1 for img in image_objs]
    return list_labels


def creat_train_test_labels():
    if Image.objects.count() != 0:
        dataset_name = get_name_dataset()

        train_imgs = Image.objects.filter(dataset_id=1)
        train_labels = creat_list_labels(train_imgs)
        train_path = path_upload_file(dataset_name + '-train-labels.npy')
        mp.save_npy_file(train_labels, train_path)

        test_imgs = Image.objects.filter(dataset_id=2)
        test_labels = creat_list_labels(test_imgs)
        test_path = path_upload_file(dataset_name + '-test-labels.npy')
        mp.save_npy_file(test_labels, test_path)
    else:
        raise ValueError('can not creat_train_test_labels, no image in data')


"""upload and predict for 1 img"""


def upload_picture(request):
    # save uploaded image by post method
    if request.method == "POST" and 'predict' not in request.POST:
        Picture.objects.all().delete()
        # Get the posted form
        form = PictureUploadForm(request.POST, request.FILES)

        if form.is_valid():
            picture = Picture()
            picture.picture = form.cleaned_data["picture"]
            picture.save()

        if Picture.objects.count() != 0:
            picture = Picture.objects.filter().latest('id')
            picture_url = picture.picture.url

            context = {
                'picture_url': picture_url,
            }

            return render(request, 'svm/upload_picture.html', context)
    else:
        form = PictureUploadForm()

    # predict for 1 uploaded image by get method
    if request.method == "POST" and 'predict' in request.POST:
        if Picture.objects.count() != 0:
            picture = Picture.objects.filter().latest('id')
            picture_url = picture.picture.url
            des = mp.feature_extract(picture_url)

            dataset_name = get_name_dataset()
            kmeans = mp.load_model_file(path_upload_file(
                dataset_name + '-kmeans.sav'))
            his = mp.create_histogram(des, kmeans)
            svm = mp.load_model_file(path_upload_file(
                dataset_name + '-svm.sav'))

            # cong 1 de giong voi index khi train tren jupyter
            label_id = svm.predict(his.reshape(1, -1)) + 1
            label = Label.objects.get(id=label_id)

            context = {
                'picture_url': picture_url,
                'label_predict': label.label_name,
                'label_description': label.description,
            }

            Picture.objects.all().delete()

            return render(request, 'svm/upload_picture.html', context)

    return render(request, 'svm/upload_picture.html')
