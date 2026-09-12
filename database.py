from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("MONGO_URI not found in .env")

client = MongoClient(MONGO_URI)

db = client["ai_lab_assistant"]

users_collection = db["users"]
chat_history_collection = db["chat_history"]
technician_records_collection = db["technician_records"]