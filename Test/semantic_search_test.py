from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embedding(text):
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=[text]
    )
    return response.data[0].embedding

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B) if norm_A and norm_B else 0.0


text1 = "python is a programming language"
text2 = "python is used heavily in data science and machine learning"
text3 = "football is a popular sport"

embedding1 = get_embedding(text1)
embedding2 = get_embedding(text2)   
embedding3 = get_embedding(text3)

similarity_1_2 = cosine_similarity(embedding1, embedding2)
similarity_1_3 = cosine_similarity(embedding1, embedding3)

print(f"Cosine similarity between text1 and text2: {similarity_1_2:.4f}")
print(f"Cosine similarity between text1 and text3: {similarity_1_3:.4f}")
