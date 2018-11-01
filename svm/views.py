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


# Create your views here.

my_dataset_path = '/media/xuantoan/Data1/CNTT/LV/Dataset/data-example'


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


"""upload and predict for 1 img"""


def upload_picture(request):
    # save uploaded image by post method
    if request.method == "POST":
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
            return render(request, 'svm/upload_picture.html')

    else:
        form = PictureUploadForm()

    # predict for 1 uploaded image by get method
    if request.method == "GET":
        if Picture.objects.count() != 0:
            picture = Picture.objects.filter().latest('id')
            picture_url = picture.picture.url
            des = mp.feature_extract(picture_url)
            loaded_kmeans = mp.load_kmeans()
            his = mp.create_histogram(des, loaded_kmeans)
            loaded_svm = mp.load_svm()

            # cong 1 de giong voi index khi train tren jupyter
            label_id = loaded_svm.predict(his.reshape(1, -1)) + 1
            label = Label.objects.get(id=label_id)

            context = {
                'picture_url': picture_url,
                'label_predict': label.label_name,
                'label_description': label.description,
            }

            Picture.objects.all().delete()

            return render(request, 'svm/upload_picture.html', context)
        else:
            return render(request, 'svm/upload_picture.html')


"""show each class 1 random img"""


def show_picture(request):
    img = Image()
    imgs = img.get_random_per_label()

    context = {
        'imgs': imgs,
    }

    return render(request, 'svm/show_picture.html', context)
