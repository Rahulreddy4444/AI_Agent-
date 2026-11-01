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
def add_embeddings(chunks):
    for chunk in chunks:
        vector_embed = get_embedding(chunk["text"])
        chunk["embedding_vector"] = vector_embed

    return chunks


# search for the top_k similar chunks to the query
def search(query, chunks, top_k=3):
    query_embed = get_embedding(query)
    similarities = []

    for chunk in chunks:
        similarity = cosine_similarity(query_embed, chunk["embedding_vector"])
        similarities.append({
            "similarity": float(similarity),
            "source": chunk["source"],
            "text": chunk["text"],
            "size": len(chunk["text"])
        })

    # sort the chunks by similarity score in descending order
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    return similarities[:top_k]


# split document into smaller chunks of text by a section using "\n\n" as a delimiter
def chunk_document(document):
    chunks = []
    sections = document["text"].split("\n\n")
    chunk_id = 0
    for section in sections:
        if section.strip():  # ignore empty sections
            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": section.strip()
            })
            chunk_id += 1

    return chunks


def create_chunks_from_documents():
    documents = load_documents()
    chunks = []

    for doc in documents:
        chunks.extend(chunk_document(doc))

    return chunks



# test the search function
def test(user_query, chunks, top_k=3):
    print(f"\nSearching for user query: '{user_query}'")

    results = search(user_query, chunks, top_k)

    print(f"\nTop {top_k} results for query: '{user_query}'")
    for i, result in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"Source: {result['source']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Size: {result['size']} characters")
        print(f"Text: {result['text'][:200]}...")  # Print first 200 characters of the text



# example usage
chunks = create_chunks_from_documents()
chunks = add_embeddings(chunks)

print(f"we have embedded {len(chunks)} chunks from the knowledge base folder: '{KB_FOLDER}'")

user_query = "what are the benefits of maternity leave?"
test(user_query, chunks, top_k=3)

