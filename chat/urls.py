from django.urls import path,include
from . import views
urlpatterns = [
    path('',views.chat,name='chat'),
    path('medico',views.medico,name='medico'),
]
