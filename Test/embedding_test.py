from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

response = client.embeddings.create(
    model=os.getenv("EMBEDDING_MODEL"),
    input = ["i am a good boy", "i like coding", "i like to play football"]
)


for i, emb in enumerate(response.data):
    print(f"Emedding {i} Length: {len(emb.embedding)}")  # Print the length of the embedding vector
    print(f"Embedding for input {i}: {emb.embedding[:10]}...")  # Print first 10 dimensions of the embedding