from pydantic import BaseModel

class PredictRequest(BaseModel):
    image: str

class GradCAMRequest(BaseModel):
    image: str
    class_index: int = -1