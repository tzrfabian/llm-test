import os
from dotenv import load_dotenv

API_KEY = "" # API key here

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")