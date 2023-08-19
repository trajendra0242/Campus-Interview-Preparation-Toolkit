from django.contrib import admin
from .models import QuestionsPython, QuestionsDS, QuestionsDBMS, QuestionsOS, QuestionsCN, Record,Solution, SolutionFace, SolutionBodyPosture, SolutionEyeContact

# Register your models here.
admin.site.register(QuestionsPython)
admin.site.register(Record)
admin.site.register(Solution)
admin.site.register(SolutionFace)
admin.site.register(SolutionBodyPosture)
admin.site.register(SolutionEyeContact)
admin.site.register(QuestionsDS)
admin.site.register(QuestionsDBMS)
admin.site.register(QuestionsOS)
admin.site.register(QuestionsCN)