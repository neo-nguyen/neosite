from django.conf.urls import url
from django.urls import path
from svm import views


urlpatterns = [
    path('', views.dataset_info, name='dataset_info'),
    path('create-labels/', views.create_labels, name='create_labels'),
    path('create-datasets/', views.create_datasets, name='create_datasets'),
    path('create-images/', views.create_images, name='create_images'),
    path('dataset-info/', views.dataset_info, name='dataset_info'),
    path('upload/', views.upload_picture, name='upload_picture'),
    path('show-picture/', views.show_picture, name='show_picture'),
    path('training/', views.training, name='training'),


]
