from django.shortcuts import render
from pathlib import Path
from svm.models import Label, Dataset, Image
from django.http import HttpResponse
import os
from django.db.models import Count
from django.db import transaction
from django.db import connection
from django.contrib import messages


# Create your views here.

my_dataset_path = '/media/xuantoan/Data1/CNTT/LV/Dataset/data-example'


def index(request):
    return HttpResponse("Welcome my svm!")


"""from a valid dataset_path
read names of label -> add to DB"""


def create_labels(dataset_path):
    train_dir = Path(dataset_path) / 'train'
    subdirs = [subdir for subdir in train_dir.iterdir() if subdir.is_dir()]
    if subdirs:
        Label.objects.all().delete()
        for subdir in subdirs:
            label = Label(label_name=subdir.parts[-1])
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
    if(request.POST.get('dataset_path')):
        dataset_path = request.POST['dataset_path']
        # validate path -> if ok ->
        read_dataset(dataset_path)
        messages.success(request, 'Read dataset successfully!')

    labels, label_nums = Label.objects.all(), Label.objects.count()
    image_nums = Image.objects.count()

    image_count = Label.objects.annotate(num_images=Count('image'))
    image_count_by_label = {}
    for i in range(len(image_count)):
        image_count_by_label[image_count[i]] = image_count[i].num_images

    context = {
        'labels': labels,
        'label_nums': label_nums,
        'image_nums': image_nums,
        'image_count_by_label': image_count_by_label,

    }

    return render(request, 'svm/dataset_info.html', context)
