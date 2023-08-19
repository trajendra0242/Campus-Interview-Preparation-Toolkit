from django.db import models
from tkinter.tix import Tree
from django.contrib.auth.models import User
from django.db import models
from django.urls.base import reverse
from embed_video.fields import EmbedVideoField
import uuid
# Create your models here.

class QuestionsPython(models.Model):
    question_number = models.IntegerField(blank=True, null=True)
    question = models.CharField(max_length=1000)
    answers = models.CharField(max_length = 5000)
    videos = models.FileField(upload_to="phase2/", blank=True)

    def __str__(self):
        return self.question

class QuestionsDS(models.Model):
    question_number = models.IntegerField(blank=True, null=True)
    question = models.CharField(max_length=1000)
    answers = models.CharField(max_length = 5000)
    videos = models.FileField(upload_to="phase2_2/", blank=True)

    def __str__(self):
        return self.question

class QuestionsDBMS(models.Model):
    question_number = models.IntegerField(blank=True, null=True)
    question = models.CharField(max_length=1000)
    answers = models.CharField(max_length = 5000)
    videos = models.FileField(upload_to="phase2_3/", blank=True)

    def __str__(self):
        return self.question

class QuestionsOS(models.Model):
    question_number = models.IntegerField(blank=True, null=True)
    question = models.CharField(max_length=1000)
    answers = models.CharField(max_length = 5000)
    videos = models.FileField(upload_to="phase2_4/", blank=True)

    def __str__(self):
        return self.question

class QuestionsCN(models.Model):
    question_number = models.IntegerField(blank=True, null=True)
    question = models.CharField(max_length=1000)
    answers = models.CharField(max_length = 5000)
    videos = models.FileField(upload_to="phase2_5/", blank=True)

    def __str__(self):
        return self.question


class Record(models.Model):
    #id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    #record_id = models.ForeignKey(QuestionsPython,on_delete= models.CASCADE,default=1)
    #record = models.FileField(upload_to="records/", blank=True)
    record_audio_text = models.CharField(max_length = 5000)
    illegal_face = models.IntegerField(blank=True, default =1)
    illegal_eye = models.IntegerField(blank=True, default =1)
    illegal_body = models.IntegerField(blank=True,default =1)
    eye_blink_count = models.IntegerField(blank=True, default =1)
    text_match = models.IntegerField(blank=True, default =1)
    emotion_report = models.CharField(max_length = 6000, default = 1)

    # def __str__(self):
    #     return self.record_id

class Solution(models.Model):

    title = models.CharField(max_length=100)
    url = EmbedVideoField()

    def __str__(self):
        return str(self.title)

class SolutionFace(models.Model):

    title = models.CharField(max_length=100)
    url = EmbedVideoField()

    def __str__(self):
        return str(self.title)

class SolutionBodyPosture(models.Model):

    title = models.CharField(max_length=100)
    url = EmbedVideoField()

    def __str__(self):
        return str(self.title)

class SolutionEyeContact(models.Model):

    title = models.CharField(max_length=100)
    url = EmbedVideoField()

    def __str__(self):
        return str(self.title)
