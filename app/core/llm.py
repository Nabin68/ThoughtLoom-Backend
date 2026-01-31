from langchain_cohere import ChatCohere
import os
from dotenv import load_dotenv

load_dotenv()
llm=ChatCohere(model="command-r-plus-08-2024")
