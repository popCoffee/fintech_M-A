import io
import os
from io import BytesIO
from django.http import HttpResponse
from django.template.loader import get_template
import pydf


class Render:

    @staticmethod
    def render(path: str, params: dict):
        template = get_template(path)
        html = template.render(params)
        response = BytesIO()
        file = open("sales_reporting.pdf", "wb")
        # pdf = pydf.generate_pdf(html)
        file.write(pdf)
        file.close()
        if not pdf.err:
            return HttpResponse(response.getvalue(), content_type='application/pdf')
        else:
            return HttpResponse("Error Rendering PDF", status=400)

    @staticmethod
    def render_to_file(path: str, params: dict):
        template = get_template(path)
        html = template.render(params)
        file_name = "sales_reporting.pdf"
        file_buffer = io.BytesIO()
        pdf = pydf.generate_pdf(html)
        file_buffer.write(pdf)
        file_buffer.seek(0)
        return [(file_name, file_buffer)]
