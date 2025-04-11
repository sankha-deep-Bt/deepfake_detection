from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_video_page, name='upload_video_page'),

]