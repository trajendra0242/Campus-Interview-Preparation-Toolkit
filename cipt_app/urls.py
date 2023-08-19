from django.urls import path
from . import views
from django.contrib.auth import views as auth_view

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('login/', auth_view.LoginView.as_view(template_name='cipt_app/login.html'), name="login"),
    path('logout/', auth_view.LogoutView.as_view(template_name='cipt_app/logout.html'), name="logout"),
    path('main/',views.main, name='main'),
    path('phase1_main/',views.phase1_main, name= 'phase1_main'),
    path('fer/',views.fer, name= 'fer'),
    path('body_posture/',views.body_posture, name= 'body_posture'),
    path('eye_detection/',views.eye_detection, name= 'eye_detection'),
]
