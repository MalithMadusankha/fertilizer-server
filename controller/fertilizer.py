from fastapi import HTTPException
from bson import ObjectId
from schemas.fertilizer import PlantFeatures
from config.db import collection
from config.recommends import growth_stages, fertilizer_recommendations
from models.model import best_model
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Predict Growth and Recommend Fertilizer
async def predict_plant_growth(features: PlantFeatures):
    try:
        sample = [[features.number_of_flowers, features.number_of_leaves, 
                   features.area_of_roots, features.number_of_stems]]

        plant_features = {
            "flowers": features.number_of_flowers,
            "leaves": features.number_of_leaves,
            "roots": features.area_of_roots,
            "stems": features.number_of_stems,
            "img_url": features.img_url
        }

        predicted_class = best_model.predict(sample)[0]

        fertilizer_info = fertilizer_recommendations.get(predicted_class, {
            "type": "Balanced (20-20-20)",
            "application_frequency": "Weekly",
            "notes": "Maintain general plant health.",
            "source": ""
        })

        # Adjust based on conditions
        if features.number_of_leaves < 3 and predicted_class > 1:
            fertilizer_info.update({
                "name": "Oasis Organic Foliar Fertilizer",
                "type": "High Nitrogen (30-10-10)",
                "application_frequency": "Every 2-3 weeks",
                "dosage": "5-10 mL per 1 liter of water for foliar spray or 1-2 teaspoons for root drenching.",
                "notes": "Encourage leaf and stem growth.",
                "source": "https://www.oasisferti.lk/"
            })

        if features.number_of_flowers < 2 and predicted_class > 3:
            fertilizer_info.update({
                "name": "Hayleys Home Garden Flower Fertilizer",
                "type": "High Phosphorus (10-30-20)",
                "application_frequency": "Bi-weekly",
                "dosage": "10 grams (1 tablespoon) per 1 liter of water for both foliar spray and root drenching.",
                "notes": "Boost flower production.",
                "source": "https://www.hayleysagriculture.com/"
            })

        if features.number_of_stems < 2 and predicted_class > 3:
            fertilizer_info.update({
                "name": "Orchid Plant Fertilizer N:P:K 12:40:12",
                "type": "High Potassium (20-20-20)",
                "application_frequency": "Weekly",
                "dosage": "1-2 grams (¼ to ½ teaspoon) per 1 liter of water for root drenching, or 1 gram per 1 liter for foliar spray.",
                "notes": "Strengthen stems and root system.",
                "source": "https://www.daraz.lk/products/orchid-plant-fertilizer-npk-124012-promotes-rapid-blooming-120g-i112554973.html"
        })


        result_data = {
            "predicted_class": int(predicted_class),
            "growth_stage": growth_stages.get(predicted_class, "Unknown stage"),
            "fertilizer_recommendation": fertilizer_info,
            "plant_features": plant_features
        }

        res = collection.insert_one(result_data)
        result_data["_id"] = str(res.inserted_id)

        return result_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get all plant growth records
async def get_all_plant_growths():
    plant_growths = list(collection.find({}))
    for record in plant_growths:
        record["_id"] = str(record["_id"])  # Convert ObjectId to string
    return {"data": plant_growths}

# Clear all plant growth records
async def clear_growths():
    result = collection.delete_many({})  # Delete all records
    return {"message": f"{result.deleted_count} plant growth records cleared."}

# Get one plant growth record by _id
async def get_plant_growth(id: str):
    try:
        object_id = ObjectId(id)
        plant_growth = collection.find_one({"_id": object_id})
        if not plant_growth:
            raise HTTPException(status_code=404, detail="Plant growth record not found")
        plant_growth["_id"] = str(plant_growth["_id"])
        return {"data": plant_growth}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {str(e)}")

# Delete one plant growth record by _id
async def delete_plant_growth(id: str):
    try:
        object_id = ObjectId(id)
        delete_result = collection.delete_one({"_id": object_id})
        if delete_result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Plant growth record not found")
        return {"message": "Plant growth record deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ID format: {str(e)}")

# clear result folder

async def clear_result():
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