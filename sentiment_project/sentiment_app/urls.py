# sentiment_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Home page
    path('analyze-text/', views.analyze_text, name='analyze_text'),  # Analyze text page
    path('dataset-analysis/', views.dataset_analysis, name='dataset_analysis'),
    path('download-csv/', views.download_csv, name='download_csv'),
]
