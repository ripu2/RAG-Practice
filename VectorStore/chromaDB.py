"""
Building a RAG System with Langchain and ChromaDB

Introduction
============
Retrieval-Augmented Generation (RAG) is a powerful technique that combines the capabilities 
of large language models with external knowledge retrieval. This script demonstrates building 
a complete RAG system using:

- LangChain: A framework for developing applications powered by language models
- ChromaDB: An open-source vector database for storing and retrieving embeddings
- OpenAI: For embeddings and language model (you can substitute with other providers)

RAG (Retrieval-Augmented Generation) Architecture:
1. Document Loading: Load documents from various sources
2. Document Splitting: Break documents into smaller chunks
3. Embedding Generation: Convert chunks into vector representations
4. Vector Storage: Store embeddings in ChromaDB
5. Query Processing: Convert user query to embedding
6. Similarity Search: Find relevant chunks from vector store
7. Context Augmentation: Combine retrieved chunks with query
8. Response Generation: LLM generates answer using context

Benefits of RAG:
- Reduces hallucinations
- Provides up-to-date information
- Allows citing sources
- Works with domain-specific knowledge
"""

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import numpy as np
from typing import List
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file")


# =============================================================================
# Step 1: Document Loading
# =============================================================================

def load_documents(path: str, glob: str) -> List[Document]:
    """Load documents from directory"""
    loader = DirectoryLoader(
        path,
        glob=glob,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    docs = loader.load()
    return docs


# =============================================================================
# Step 2: Split Documents into Chunks
# =============================================================================

def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    
    # Filter out chunks that are too short (likely just titles)
    MIN_CHUNK_LENGTH = 100
    filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content) >= MIN_CHUNK_LENGTH]
    
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    print(f"Filtered to {len(filtered_chunks)} chunks (removed {len(chunks) - len(filtered_chunks)} short chunks)")
    
    # Show chunk details
    for i, chunk in enumerate(filtered_chunks, 1):
        preview = chunk.page_content[:60].replace('\n', ' ') + "..."
        print(f"  Chunk {i}: {len(chunk.page_content)} chars - {preview}")
    
    return filtered_chunks


# =============================================================================
# Step 3: Create Embeddings
# =============================================================================

def get_embeddings() -> OpenAIEmbeddings:
    """Initialize OpenAI embeddings"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


# =============================================================================
# Step 4: Create ChromaDB Vector Store
# =============================================================================

def create_vector_store(chunks: List[Document], embeddings: OpenAIEmbeddings) -> Chroma:
    """Create and persist ChromaDB vector store"""
    import shutil
    from pathlib import Path
    
    persist_directory = "./chroma_db"
    
    # Clean up any existing database to avoid conflicts
    if Path(persist_directory).exists():
        print(f"Removing existing database at {persist_directory}...")
        shutil.rmtree(persist_directory)
    
    print(f"Creating new vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="rag_collection"
    )
    
    print(f"✓ Created vector store with {len(chunks)} chunks")
    print(f"✓ Persisted to: {persist_directory}")
    
    return vector_store


# =============================================================================
# Complete RAG Pipeline
# =============================================================================

def process_and_store_documents(data_path: str = None) -> Chroma:
    """Complete pipeline: Load -> Split -> Embed -> Store"""
    
    # Step 1: Load documents
    print("Step 1: Loading documents...")
    
    # If no path provided, use path relative to this script's location
    if data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, '../data/text')
    
    docs = load_documents(data_path, 'rag_example*')
    print(f"Loaded {len(docs)} documents\n")
    
    # Step 2: Split into chunks
    print("Step 2: Splitting documents...")
    chunks = split_documents(docs)
    print()
    
    # Step 3: Initialize embeddings
    print("Step 3: Initializing embeddings...")
    embeddings = get_embeddings()
    print("Embeddings initialized\n")
    
    # Step 4: Create vector store
    print("Step 4: Creating vector store...")
    vector_store = create_vector_store(chunks, embeddings)
    
    return vector_store


# =============================================================================
# Step 5: Query and Search Functions
# =============================================================================

def search_with_scores(query: str, vector_store: Chroma, k: int = 3):
    """Search vector store and return results with similarity scores
    
    Lower scores = more similar (distance-based)
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


def display_search_results(query: str, results):
    """Display search results in a formatted way"""
    print(f"\n{'='*80}")
    print(f"🔍 Query: '{query}'")
    print(f"{'='*80}\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"📄 Result #{i}")
        print(f"   Similarity Score: {score:.4f} (lower is better)")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Length: {len(doc.page_content)} characters")

        print(f"\n   Content Preview:")
        print(f"   {'-'*76}")
        # Show first 300 chars or full content if shorter
        preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        for line in preview.split('\n'):
            print(f"   {line}")
        print(f"   {'-'*76}\n")


"""
Understanding Similarity Scores
================================
ChromaDB returns distance scores (L2/Euclidean distance by default):
- Lower scores = More similar (closer in vector space)
- Higher scores = Less similar (farther apart)

General Guidelines:
- 0.0 - 0.5: Highly similar (excellent match)
- 0.5 - 1.0: Moderately similar (good match)
- 1.0 - 1.5: Somewhat similar (acceptable match)
- > 1.5: Low similarity (may not be relevant)

Note: Exact thresholds depend on:
- Your embedding model (text-embedding-3-small in our case)
- Document length and complexity
- Query specificity
"""


def interpret_score(score: float) -> tuple[str, str]:
    """Interpret similarity score and return quality rating and emoji"""
    if score < 0.5:
        return "Excellent", "🟢"
    elif score < 1.0:
        return "Good", "🟡"
    elif score < 1.5:
        return "Fair", "🟠"
    else:
        return "Poor", "🔴"


def filter_by_threshold(results, threshold: float = 1.5):
    """Filter results by similarity threshold"""
    filtered = [(doc, score) for doc, score in results if score <= threshold]
    print(f"Filtered {len(results)} → {len(filtered)} results (threshold: {threshold})")
    return filtered


def display_search_results_with_quality(query: str, results):
    """Display search results with quality indicators"""
    print(f"\n{'='*80}")
    print(f"🔍 Query: '{query}'")
    print(f"{'='*80}\n")
    
    for i, (doc, score) in enumerate(results, 1):
        quality, emoji = interpret_score(score)
        
        print(f"{emoji} Result #{i} - {quality} Match")
        print(f"   Score: {score:.4f} (lower is better)")
        print(f"   Source: {doc.metadata['source']}")
        print(f"   Length: {len(doc.page_content)} chars")
        print(f"\n   Content Preview:")
        print(f"   {'-'*76}")
        # Show first 250 chars
        preview = doc.page_content[:250]
        for line in preview.split('\n'):
            print(f"   {line}")
        if len(doc.page_content) > 250:
            print(f"   ... ({len(doc.page_content) - 250} more chars)")
        print(f"   {'-'*76}\n")


def search_vector_store(query: str, vector_store: Chroma) -> List[Document]:
    """Search vector store for similar documents"""
    return vector_store.similarity_search(query, k=3)


"""
How to Use Similarity Scores in Practice
=========================================

1. Setting Thresholds:
   # Only return highly relevant results
   good_results = filter_by_threshold(results, threshold=0.8)

2. Contextual Quality Check:
   - If all results have scores > 1.5: Query might be too vague or documents don't cover the topic
   - If top result has score < 0.3: Excellent match, high confidence
   - If scores are similar (e.g., 0.65, 0.68, 0.71): Multiple relevant documents

3. Best Practices:
   - Development: Set threshold ~1.2 to see what gets filtered
   - Production: Set threshold 0.8-1.0 for quality results
   - Always return top result if score < 1.5 (likely relevant)
   - Log scores to understand your system's typical ranges

4. Improving Low Scores:
   - Rephrase the query to be more specific
   - Add more context to documents
   - Adjust chunk size (larger chunks = more context)
   - Try different embedding models
"""


# =============================================================================
# Initialize LLM
# =============================================================================

def initialize_llm():
    """Initialize the language model"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=500)
    return llm


"""
LLM Provider Comparison
=======================

OpenAI (gpt-4o-mini)
- Already configured (OPENAI_API_KEY in .env)
- High quality, well-tested
- Cost: ~$0.15 per 1M input tokens
- Best for: Production RAG systems

Groq (llama-3.1-8b-instant)
- Requires GROQ_API_KEY in .env file
- Extremely fast inference (fastest LLM API)
- Cost: Often free tier available
- Best for: Fast prototyping, high throughput
- Popular models:
  - llama-3.1-8b-instant: Fast, good quality
  - llama-3.1-70b-versatile: Better quality, slower
  - mixtral-8x7b-32768: Large context window

To use Groq:
1. Get API key from https://console.groq.com
2. Add to .env: GROQ_API_KEY=your_key_here
3. Use init_chat_model("llama-3.1-8b-instant", model_provider="groq") instead
"""


# =============================================================================
# Create RAG Chain
# =============================================================================

def create_rag_chain(vector_store: Chroma, llm):
    """Create a complete RAG chain"""
    
    # Convert vector store to retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    
    # Prompt Template
    system_prompt = """ You are a helpful assistant that can answer questions based on the context provided.
Use the below  piece of retrieved context to answer the question.
If the context is not relevant to the question, say "I don't know"

Use maximum 3 sentences to answer the question.

Context: {context}
"""
    
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
        ]
    )
    
    # Create a document chain
    # Takes Retrieved Doc -> Stuff the context placeholder -> send to llm -> return llm response
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )
    
    # Create final RAG chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    return rag_chain


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the complete RAG system"""
    
    # Run the pipeline to create vector store
    print("="*80)
    print("BUILDING RAG SYSTEM")
    print("="*80 + "\n")
    
    vector_store = process_and_store_documents()
    
    # Test similarity search
    print("\n" + "="*80)
    print("TESTING SIMILARITY SEARCH")
    print("="*80)
    
    query = "What is deep learning?"
    results = search_with_scores(query, vector_store, k=3)
    display_search_results(query, results)
    
    # Test different queries
    print("\n" + "="*80)
    print("COMPARING SIMILARITY SCORES ACROSS QUERIES")
    print("="*80)
    
    queries = [
        "What is deep learning?",
        "Explain neural networks",
        "What are the types of machine learning?",
        "Tell me about NLP applications"
    ]
    
    for query in queries:
        results = search_with_scores(query, vector_store, k=3)
        top_score = results[0][1] if results else float('inf')
        quality, emoji = interpret_score(top_score)
        
        print(f"\n{emoji} Query: '{query}'")
        print(f"   Best Score: {top_score:.4f} ({quality})")
        print(f"   Top Result: {results[0][0].page_content[:80].replace(chr(10), ' ')}...")
        print(f"   {'-'*76}")
    
    # Initialize LLM and create RAG chain
    print("\n" + "="*80)
    print("INITIALIZING LLM AND RAG CHAIN")
    print("="*80 + "\n")
    
    llm = initialize_llm()
    rag_chain = create_rag_chain(vector_store, llm)
    
    # Test RAG chain
    print("Testing RAG chain with query: 'What is deep learning?'\n")
    response = rag_chain.invoke({"input": "What is deep learning?"})
    print(f"Answer: {response['answer']}")
    
    print("\n" + "="*80)
    print("RAG SYSTEM READY")
    print("="*80)


if __name__ == "__main__":
    main()
