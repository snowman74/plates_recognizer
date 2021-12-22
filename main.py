from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
from variables import *
from functions import final_detect
from fastapi.responses import StreamingResponse

app = FastAPI(title="Нейронка макса")


@app.post('/detection/')
async def image_recognition(image: UploadFile = File(...)):
    np_image = np.array(Image.open(io.BytesIO(await image.read())))
    result_image, plate_name = final_detect(np_image, model_plates, model_chars, font_path)
    my_img = Image.fromarray(result_image)
    buffer = io.BytesIO()
    my_img.save(buffer, format='JPEG')
    buffer.seek(0)
    return StreamingResponse(buffer, media_type='image/jpg')
