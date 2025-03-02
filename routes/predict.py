from fastapi import APIRouter
from models.activityModel import ActivityModel
from models.gameModel import GameModel
from schemas.serialize import serializeDict, serializeList
from controller.predictor import ActivityPredictor, GamePredictor

predict = APIRouter()


@predict.post('/predict-activity')
async def activity_predict(activityObj: ActivityModel):
    res = ActivityPredictor(activityObj)
    return res


@predict.post('/predict-game')
async def game_predict(gameObj: GameModel):
    print("========= ", gameObj)
    res = GamePredictor(gameObj)
    return res
