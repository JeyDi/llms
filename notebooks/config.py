import os
import openai
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
# client = OpenAI()
# client.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]
llm_model = "gpt-4"