from fastapi import APIRouter
from controller.fertilizer import get_plant_growth, delete_plant_growth, get_all_plant_growths, clear_result, predict_plant_growth, clear_growths
from schemas.fertilizer import PlantFeatures

import os
import uuid
import math
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from models.model import yolo_model

fertilizer = APIRouter()

# Create necessary directories
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

@fertilizer.post("/features")
async def upload_images(
    file1: UploadFile = File(...), 
    file2: UploadFile = File(...), 
    file3: UploadFile = File(...)
):
    print("<=================== features ===================>")

    # Validate files
    for file in [file1, file2, file3]:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="All files must be images")

    total_counts = {class_name: 0 for class_name in yolo_model.names.values()}
    num_images = 3
    last_result_image_path = None

    for file in [file1, file2, file3]:
        file_id = str(uuid.uuid4())
        temp_file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            results = yolo_model(temp_file_path)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls.cpu().numpy()[0])
                    class_name = yolo_model.names[class_id]
                    total_counts[class_name] += 1

                last_result_image_path = RESULTS_DIR / f"result_{file_id}.jpg"
                result.save(str(last_result_image_path))
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        
        finally:
            if temp_file_path.exists():
                os.remove(temp_file_path)

    average_counts = {class_name: math.ceil(count / num_images) for class_name, count in total_counts.items()}

    result_image_url = f"/results/{last_result_image_path.name}" if last_result_image_path else None

    return {
        "status": "success",
        "message": "Images processed successfully",
        "class_counts": average_counts,
        "result_image_url": result_image_url
    }


@fertilizer.post("/plant-growth")
async def predict_growth(features: PlantFeatures):
    return await predict_plant_growth(features)

# Fetch all plant growth records
@fertilizer.get("/plant-growths")
async def fetch_all():
    return await get_all_plant_growths()

# Fetch a single record by _id
@fertilizer.get("/plant-growth/{id}")
async def fetch_one(id: str):
    return await get_plant_growth(id)

# Delete a record by _id
@fertilizer.delete("/plant-growth/{id}")
async def delete_one(id: str):
    return await delete_plant_growth(id)


@fertilizer.delete("/clear-results")
async def clear_results():
    print("<=================== clear-results ===================>")
    return await clear_result()

@fertilizer.delete("/delete-growths")
async def clear_results():
    print("<=================== clear-results ===================>")
    return await clear_growths()