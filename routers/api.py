import base64
import io
import numpy as np
from PIL import Image
from fastapi import UploadFile, File
from fastapi.routing import APIRouter
from tools.functions import final_detect
from tools.variables import *
from pydantic_models import OutputNeural


router = APIRouter(prefix='/api',
                   tags=['api'])


@router.post('/detection/', response_model=OutputNeural)
async def detection(image: UploadFile = File(...)):
    buffer = io.BytesIO()
    np_image = np.array(Image.open(io.BytesIO(await image.read())))
    result_image, plate_name = final_detect(np_image, MODEL_PLATES, MODEL_CHARS, FONT_PATH)
    result_image = Image.fromarray(result_image)
    result_image.save(buffer, format='JPEG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {'plate': plate_name, 'img_str': img_str}

