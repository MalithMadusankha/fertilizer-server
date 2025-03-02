from pydantic import BaseModel
from bson import ObjectId


class GameModel(BaseModel):
    tryG: int
    duration1: int
    status1: int
    level1: int
    duration2: int
    status2: int
    level2: int
    duration3: int
    status3: int
    level3: int
