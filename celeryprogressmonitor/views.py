import json
from django.http import HttpResponse, JsonResponse
from celery_progress.backend import Progress
import decimal


def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


def get_celery_task_progress(request):
    task_id = str(request.GET['task_id'])
    progress = Progress(task_id)
    return HttpResponse(json.dumps(progress.get_info(), default=decimal_default), content_type='application/json')


def get_multiple_tasks_progress(request):
    task_progress_response = {}
    try:
        if 'task_id_dict' in request.GET:
            task_id_dict = json.loads(request.GET['task_id_dict'])
            for each_fund in task_id_dict:
                each_task_id = task_id_dict[each_fund]
                task_progress_response[each_task_id] = Progress(each_task_id).get_info()
                task_progress_response[each_task_id]['fund'] = each_fund
    except KeyError:
        task_progress_response = 'Key-error Failure'

    return JsonResponse(task_progress_response)
