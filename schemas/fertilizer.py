from pydantic import BaseModel

class PlantFeatures(BaseModel):
    number_of_flowers: int
    number_of_leaves: int
    area_of_roots: int  # in pixels
    number_of_stems: int
    img_url: str
