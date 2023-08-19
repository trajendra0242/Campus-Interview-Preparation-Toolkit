from django.urls import path
from . import views

urlpatterns = [
    path('', views.tech_int_home, name='tech_int_home'),
    path('tech_int_select/', views.tech_int_select, name='tech_int_select'),
    path('tech_int_python/', views.tech_int_python, name='tech_int_python'),
    path('tech_int_python/', views.tech_int_python, name='tech_int_ds'),
    path('tech_int_python/', views.tech_int_python, name='tech_int_dbms'),
    path('tech_int_python/', views.tech_int_python, name='tech_int_os'),
    path('tech_int_python/', views.tech_int_python, name='tech_int_cn'),

    path('record/', views.record, name='record'),
    path('report/', views.report, name='report'),
    path('illegal_face/', views.illegal_face, name='illegal_face'),
    path('illegal_body/', views.illegal_body, name='illegal_body'),
    path('illegal_eye/', views.illegal_eye, name='illegal_eye'),
    path('tech_int_issue/', views.tech_int_issue, name='tech_int_issue'),

]
