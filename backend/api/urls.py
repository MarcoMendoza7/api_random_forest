# api/urls.py
from django.urls import path
from .views import train_rf  # importamos solo la función que existe

urlpatterns = [
    path('train_rf/', train_rf, name='train_rf'),
]
