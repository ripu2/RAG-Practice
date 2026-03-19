# Hugging face Embeddings

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Create a vector for a sentence
sentence = "I love programming in Python"
vector = embeddings.embed_query(sentence)
print(len(vector))
