from pydantic import BaseModel


class OutputNeural(BaseModel):
    plate: list
    img_str: str
