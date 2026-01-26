from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from v2.utils.config import getEnvValue

MONGODB_URI = getEnvValue('MONGODB_URI')

client = MongoClient(MONGODB_URI)

memory = MongoDBSaver(client)
