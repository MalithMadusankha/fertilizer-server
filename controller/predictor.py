import os
import pandas as pd
import numpy as np
import joblib
from models.activityModel import ActivityModel
from models.gameModel import GameModel
from fastapi.encoders import jsonable_encoder

# Load the model from the saved file
ActivityDTCM = joblib.load('controller/ActivityDTCM.joblib')
GameDTCM = joblib.load('controller/GameDTCM.joblib')


current_path = os.getcwd()


def ActivityPredictor(activityObj: ActivityModel):

    print('<<================== Activity Predictor ==================>>')

    data = {
        'Age': activityObj.age,
        'Activity_Count': activityObj.activity_count,
        'Breakfast_Time': activityObj.breakfast_time,
        'M_Play_Time': activityObj.m_play_time,
        'Lunch_Time': activityObj.lunch_time,
        'Clean_Time': activityObj.clean_time,
        'E_Play_Time': activityObj.e_play_time,
        'Bath_Time': activityObj.bath_time,
        'Dinner_Time': activityObj.dinner_time
    }
    df = pd.DataFrame(data)
    print(df)

    sample = df.values

    predict_res = ActivityDTCM.predict(sample)
    print('predic', predict_res)

    recommen = ''

    if (predict_res[0] == 1):
        recommen = 'Poor, Child need more attention, follow same schedule'
    elif predict_res[0] == 2:
        recommen = 'Fair, Child Need attention, try to train again'
    elif predict_res[0] == 3:
        recommen = 'Good, try again and again, try to keep force'
    elif predict_res[0] == 4:
        recommen = 'Very Good, give similar activities '
    else:
        recommen = 'Excellent, give similar and different types of activities'

    res = {"predict_res": int(predict_res[0]), "recommen": recommen}
    response_content = jsonable_encoder(res)

    return response_content


def GamePredictor(gameObj: GameModel):
    print('<<================== Game Predictor ==================>>')

    s2 = [[gameObj.tryG], [gameObj.duration1], [gameObj.status1], [gameObj.level1], [gameObj.duration2], [gameObj.status2], [
        gameObj.level2], [gameObj.duration3], [gameObj.status3], [gameObj.level3]]
    df = pd.DataFrame({'Try': s2[0], 'Duration1': s2[1], 'Status1': s2[2],
                       'Level1': s2[3], 'Duration2': s2[4], 'Status2': s2[5],
                       'Level2': s2[6], 'Duration3': s2[7], 'Status3': s2[8],
                       'Level3': s2[9]})
    print(df)

    sample = df.iloc[:, :].values

    predict_res = GameDTCM.predict(sample)
    print('predic', predict_res)

    recommen = ''

    if (predict_res[0] == 1):
        recommen = 'Poor, Child need more attention, follow the same game'
    elif predict_res[0] == 2:
        recommen = 'Fair, Child Need attention, try to play game again and again'
    elif predict_res[0] == 3:
        recommen = 'Good, try again and again, try to keep force'
    elif predict_res[0] == 4:
        recommen = 'Very Good, give similar games'
    else:
        recommen = 'Execellent, give similar and differnt types of games'

    res = {"predict_res": int(predict_res[0]), "recommen": recommen}
    response_content = jsonable_encoder(res)

    return response_content
