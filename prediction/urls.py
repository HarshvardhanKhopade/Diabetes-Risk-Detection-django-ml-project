from django.urls import path
from .views import predict_diabetes, predict_diabetes_api

urlpatterns = [
    path('', predict_diabetes, name='predict_diabetes'),
    path('api/predict/', predict_diabetes_api, name='predict_diabetes_api'),
]
