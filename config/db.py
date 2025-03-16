from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://dbuser:malith123@clusterdev.uocaz.mongodb.net/") # Update with your MongoDB connection string if needed
db = client["OrchidFertilizerDB"]  # Database name
collection = db["FertilizerRecommendations"]  # Collection name


