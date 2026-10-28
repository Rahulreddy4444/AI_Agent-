import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

clinet = OpenAI(
    base_url=os.getenv("BASE_URL"), 
    api_key=os.getenv("API_KEY")
)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
KB_FOLDER = os.getenv("KB_FOLDER")

# load the documentation files from the knowledge base folder
def load_documents(folder=KB_FOLDER):
    documents = []

    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents.append(
                    {
                        "source": filename,
                        "text": f.read()
                    }
                )

    return documents



# get embedding for a given text
def get_embedding(text):
    response = clinet.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return response.data[0].embedding


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B) if norm_A and norm_B else 0.0


# add embedding to the documents
def add_embeddings(documents):
    for doc in documents:
        vector_embed = get_embedding(doc["text"])
        doc["embedding_vector"] = vector_embed

    return documents


# search for the top_k similar document to the query
def search(query, documents, top_k=3):
    query_embed = get_embedding(query)
    similarities = []

    for doc in documents:
        similarity = cosine_similarity(query_embed, doc["embedding_vector"])
        similarities.append({
            "similarity": float(similarity),
            "source": doc["source"],
            "text": doc["text"],
            "size": len(doc["text"])
        })

    # sort the documents by similarity score in descending order
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return similarities[:top_k]



# test the search function
def test(user_query, documents, top_k=3):
    print(f"\nSearching for user query: '{user_query}'")

    results = search(user_query, documents, top_k)

    print(f"\nTop {top_k} results for query: '{user_query}'")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Source: {result['source']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Size: {result['size']} characters")
        print(f"Text: {result['text'][:200]}...")  # Print first 200 characters of the text



# example usage
documents = load_documents()
documents = add_embeddings(documents)

print(f"we have embedded {len(documents)} documents from the knowledge base folder: '{KB_FOLDER}'")

user_query = "How many days i will get for maternity leave?"
test(user_query, documents, top_k=3)

