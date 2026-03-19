# Open AI Embedding

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPEN_AI_API_KEY"))

sentence = "I love programming in Python"
vector = embeddings.embed_query(sentence)
print(len(vector))

# Cosine Similarity

import numpy as np

def cosine_similarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_similarity = numerator / denominator
    if cosine_similarity > 0.9:
        return "Very similar"
    elif cosine_similarity > 0.5:
        return "Similar"
    elif cosine_similarity > 0.1:
        return "Not similar"
    else:
        return "Opposite"

sentences = [
    "I love programming in Python",
    "My buddy is the best dog in the world",
    "My buddy is good",
    "I am a good person",
    "A cat sat on my bed",
    "My Buddy Sleeps with me",
    "My Buddy is a bad"
]

sentence_embeddings = embeddings.embed_documents(sentences)

for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        print(f"For the sentence {sentences[i]} and {sentences[j]}, the similarity is {cosine_similarity(sentence_embeddings[i], sentence_embeddings[j])}")

# Semantic Search

from enum import Enum

class EmbeddingType(Enum):
    SENTENCE = "sentence"
    DOCUMENT = "document"

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPEN_AI_API_KEY"))

def generateEmbeddings(type: EmbeddingType, text: str | list[str]):
    """
    Generate embeddings for either:
    - SENTENCE: Single string query (returns single embedding vector)
    - DOCUMENT: List of strings (returns list of embedding vectors)
    """
    try:
        if type == EmbeddingType.SENTENCE:
            if isinstance(text, str) and text:
                return embeddings.embed_query(text)
            else:
                raise ValueError("SENTENCE type requires a non-empty string")
                
        elif type == EmbeddingType.DOCUMENT:
            if isinstance(text, list) and len(text) > 0:
                return embeddings.embed_documents(text)
            else:
                raise ValueError("DOCUMENT type requires a non-empty list of strings")
        else: 
            raise ValueError("Invalid embedding type")
            
    except Exception as e:
        print(f"Error: {e}")
        return None

documents = [
    "LangChain is a framework for developing applications powered by large language models",
    "Application powered by large language models are called as LLMs use Python language",
    "Python is a high-level programming language",
    "Machine learning is a subset of artificial intelligence",
    "Embeddings convert text into numerical vectors",
    "The weather today is sunny and warm"
]

import numpy as np

def cosine_similarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_similarity = numerator / denominator
    return cosine_similarity

def calculate_similarity(query_embedding, document_embeddings):
    similarities = []
    for document_embedding in document_embeddings:
        similarity = cosine_similarity(query_embedding, document_embedding)
        similarities.append(similarity)
    
    return similarities

def generateResponse(query: str):
    query_embedding = generateEmbeddings(EmbeddingType.SENTENCE, query)

    document_embeddings = generateEmbeddings(EmbeddingType.DOCUMENT, documents)

    similarities = calculate_similarity(query_embedding, document_embeddings)

    max_similarity_index = similarities.index(max(similarities))
    print(f"Query: {query} \nAnswer: {documents[max_similarity_index]}")

query = "What is subset of artificial intelligence?"

generateResponse(query)

query = "What is embeddings?"

generateResponse(query)

query = "Which language is used in high level programming?"

generateResponse(query)
