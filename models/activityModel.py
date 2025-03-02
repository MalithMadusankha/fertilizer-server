from pydantic import BaseModel
from bson import ObjectId
from typing import List


class ActivityModel(BaseModel):
    age: List[int]
    activity_count: List[int]
    breakfast_time: List[int]
    m_play_time: List[int]
    lunch_time: List[int]
    clean_time: List[int]
    e_play_time: List[int]
    bath_time: List[int]
    dinner_time: List[int]
