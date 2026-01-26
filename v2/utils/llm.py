import os

from langchain.chat_models import init_chat_model
from v2.utils.config import getEnvValue

os.environ['GROQ_API_KEY'] = getEnvValue('GROQ_API_KEY')
model = getEnvValue('LLM_MODEL')

llm = init_chat_model(model, model_provider="groq", temperature=0)