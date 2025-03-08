import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import shutil
from tempfile import NamedTemporaryFile
from pathlib import Path
from ultralytics import YOLO
import uuid
import joblib
from pydantic import BaseModel

app = FastAPI()

# Set up CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://*.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path_r = os.getcwd()
# print("Current path",path_r)
# Load the YOLO model - update this path to the location in your deployed app
MODEL_PATH = path_r + "/models/orchid_yolo_model.pt"  # Adjust this path as needed
model = YOLO(MODEL_PATH)

# Create temp directory for uploaded images
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load the trained DecisionTreeClassifier model
GROWTH_MODEL_PATH = path_r + "/models/orchid_growth_model.pkl"
best_model = joblib.load(GROWTH_MODEL_PATH)

# Define request body structure
class PlantFeatures(BaseModel):
    number_of_flowers: int
    number_of_leaves: int
    area_of_roots: int  # in pixels
    number_of_stems: int

@app.get("/")
async def read_root():
    print("<========= Call Default route ==========>")
    return {"message": "Hello !!!, I am FastAPI Server. U can call my API I am here to respond"}

@app.post("/features")
async def predict_image(file: UploadFile = File(...)):
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Create unique filename
    file_id = str(uuid.uuid4())
    temp_file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run inference on the image
        results = model(temp_file_path)
        
        # Process results
        detection_results = []
        class_counts = {class_name: 0 for class_name in model.names.values()}  # Initialize counts for each class
        
        for result in results:
            # Save the result image
            result_file_path = RESULTS_DIR / f"result_{file_id}.jpg"
            result.save(str(result_file_path))
            
            # Extract detection data
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = model.names[class_id]
                class_counts[class_name] += 1  # Increment the count for the detected class
                
        # Return the result
        return {
            "status": "success",
            "message": "Image processed successfully",
            "class_counts": class_counts,  # Add the class counts to the response
            "result_image_url": f"/results/{result_file_path.name}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if temp_file_path.exists():
            os.remove(temp_file_path)

@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Result image not found")
    return FileResponse(str(file_path))


@app.delete("/clear-results")
async def clear_results():
    try:
        # Delete all files in the results directory
        for file in RESULTS_DIR.iterdir():
            if file.is_file():
                file.unlink()  # Delete the file
        
        return {
            "status": "success",
            "message": "Results folder cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing results folder: {str(e)}")

@app.post("/plant-growth")
async def predict_plant_growth(features: PlantFeatures):
    # Convert input data into a list format
    sample = [[
        features.number_of_flowers,
        features.number_of_leaves,
        features.area_of_roots,
        features.number_of_stems
    ]]
    
    # Predict plant age category
    predicted_class = best_model.predict(sample)[0]

    # Growth stages mapping
    growth_stages = {
        1: "Baby plant (weeks 0 - 24)",
        2: "Child plant (weeks 25 - 52)",
        3: "Young plant (weeks 53 - 104)",
        4: "Mature plant (weeks 105 - 156)",
        5: "Adult plant (more than 157 weeks)"
    }

    # Predefined fertilizer recommendations
    fertilizer_recommendations = {
        1: {
            "name": "Oasis Organic Foliar Fertilizer",
            "type": "Organic foliar fertilizer",
            "application_frequency": "Every 2-3 weeks",
            "notes": "Supports initial growth and leaf development.",
            "source": "https://www.oasisferti.lk/"
        },
        2: {
            "name": "Hayleys Home Garden Flower Fertilizer",
            "type": "Balanced fertilizer",
            "application_frequency": "Bi-weekly",
            "notes": "Encourages foliar and root development.",
            "source": "https://www.hayleysagriculture.com/crop-solutions/floriculture-cultivators/growing-orchids-in-sri-lanka/"
        },
        3: {
            "name": "Bio Foods Certified Organic Fertilizer",
            "type": "Organic fertilizer",
            "application_frequency": "Weekly during active growth periods",
            "notes": "Promotes vigorous vegetative growth.",
            "source": "https://www.biofoodslk.com/fertilizer"
        },
        4: {
            "name": "Daraz Garden Soil & Fertilizers",
            "type": "Bloom-boosting fertilizer",
            "application_frequency": "Weekly during blooming season",
            "notes": "Enhances flower production and quality.",
            "source": "https://www.daraz.lk/soils-fertilisers-mulches/"
        },
        5: {
            "name": "Oasis Organic Foliar Fertilizer",
            "type": "Balanced fertilizer",
            "application_frequency": "Weekly, with occasional plain water flushes to prevent salt buildup",
            "notes": "Maintains overall plant health and supports continuous blooming cycles.",
            "source": "https://www.oasisferti.lk/"
        }
    }

    # **Dynamic Fertilizer Recommendation Based on Features**
    fertilizer_info = fertilizer_recommendations.get(predicted_class, {
        "type": "Balanced (20-20-20)",
        "application_frequency": "Weekly",
        "notes": "Maintain general plant health.",
        "source": ""
    })

    # Adjust fertilizer based on feature analysis
    if features.number_of_leaves < 3 and predicted_class > 1:
        fertilizer_info["type"] = "High Nitrogen (30-10-10)"
        fertilizer_info["name"] = "Oasis Organic Foliar Fertilizer"
        fertilizer_info["application_frequency"] = "Every 2-3 weeks"
        fertilizer_info["notes"] = "Encourage leaf and stem growth."
        fertilizer_info["source"] = "https://www.oasisferti.lk/"

    if features.number_of_flowers < 2 and predicted_class > 3:
        fertilizer_info["type"] = "High Phosphorus (10-30-20)"
        fertilizer_info["name"] = "Hayleys Home Garden Flower Fertilizer"
        fertilizer_info["application_frequency"] = "Bi-weekly"
        fertilizer_info["notes"] = "Boost flower production."
        fertilizer_info["source"] = "https://www.hayleysagriculture.com/crop-solutions/floriculture-cultivators/growing-orchids-in-sri-lanka/"

    if features.number_of_stems < 2 and predicted_class > 3:
        fertilizer_info["type"] = "High Potassium (20-20-20)"
        fertilizer_info["name"] = "Bio Foods Certified Organic Fertilizer"
        fertilizer_info["application_frequency"] = "Weekly during active growth periods"
        fertilizer_info["notes"] = "Strengthen stems and root system."
        fertilizer_info["source"] = "https://www.biofoodslk.com/fertilizer"

    # For general growth and flowering support:
    if features.number_of_leaves >= 3 and features.number_of_flowers >= 2 and features.number_of_stems >= 2:
        fertilizer_info["type"] = "Balanced (20-20-20)"
        fertilizer_info["name"] = "Daraz Garden Soil & Fertilizers"
        fertilizer_info["application_frequency"] = "Weekly during blooming season"
        fertilizer_info["notes"] = "Supports overall plant health, encourages continuous blooming."
        fertilizer_info["source"] = "https://www.daraz.lk/soils-fertilisers-mulches/"

    return {
        "predicted_class": int(predicted_class),
        "growth_stage": growth_stages.get(predicted_class, "Unknown stage"),
        "fertilizer_recommendation": fertilizer_info
    }

print("<============== Server started ==============>")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)