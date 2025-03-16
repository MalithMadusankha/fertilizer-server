import joblib
import os
from ultralytics import YOLO

model_path_yolo = os.path.join(os.path.dirname(__file__), "orchid_yolo_model.pt")
yolo_model = YOLO(model_path_yolo)

model_path = os.path.join(os.path.dirname(__file__), "orchid_growth_modeln.pkl")
best_model = joblib.load(model_path)
