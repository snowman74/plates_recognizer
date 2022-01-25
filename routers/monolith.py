import base64
import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates
from tools.functions import final_detect
from tools.variables import *


templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix='/monolith',
                   tags=['monolith'])


@router.get('/home/', response_class=HTMLResponse)
async def hello_world(request: Request):
    return templates.TemplateResponse('home.html', {'request': request,
                                                    'result': None})


@router.post('/home/', response_class=HTMLResponse)
async def form_post(request: Request, image: UploadFile = File(...)):
    buffer = io.BytesIO()
    np_image = np.array(Image.open(io.BytesIO(await image.read())))
    result_image, plate_name = final_detect(np_image, MODEL_PLATES, MODEL_CHARS, FONT_PATH)
    result_image = Image.fromarray(result_image)
    result_image.save(buffer, format='JPEG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return templates.TemplateResponse('home.html', {'request': request,
                                                    'result': img_str})

