from django.conf.urls import url
from . import views

app_name = 'celeryprogressmonitors'
urlpatterns = [
    url('get_celery_task_progress$', views.get_celery_task_progress, name='get_celery_task_progress'),
    url('get_multiple_tasks_progress$', views.get_multiple_tasks_progress, name='get_multiple_tasks_progress'),
]
