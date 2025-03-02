from pymongo import MongoClient
from pymongo.encryption import ClientEncryption
conn = MongoClient("mongodb://localhost:27017/nest")
