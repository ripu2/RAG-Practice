# Complete Embeddings Pipeline
# This file imports and demonstrates all embedding functionality

print("="*60)
print("EMBEDDINGS PIPELINE")
print("="*60)

print("""
This directory contains implementations for:

1. embeddings_info.py
   - Concepts and theory of embeddings
   - 2D and 3D visualizations
   - Cosine similarity examples
   - Feature representation demonstrations

2. openai_embeddings.py
   - OpenAI text-embedding-3-small model
   - 1536-dimensional vectors
   - Cosine similarity implementation
   - Semantic search demo
   - Requires API key

3. huggingface_embeddings.py
   - HuggingFace sentence-transformers model
   - 384-dimensional vectors
   - Free and open-source
   - Runs locally (no API calls)

USAGE:
------
Run individual files to see specific demos:
  python embeddings_info.py          # Theory and visualizations
  python openai_embeddings.py        # OpenAI embeddings
  python huggingface_embeddings.py   # HuggingFace embeddings

COMPARISON:
-----------
OpenAI Embeddings:
  ✓ High quality (1536 dimensions)
  ✓ State-of-the-art performance
  ✗ Requires API key and costs money
  ✗ Needs internet connection

HuggingFace Embeddings:
  ✓ Free and open-source
  ✓ Runs locally
  ✓ Good performance (384 dimensions)
  ✓ No API calls needed
  ✗ Slightly lower quality than OpenAI

WHEN TO USE WHICH:
------------------
- Use OpenAI for: Production apps, highest quality needs
- Use HuggingFace for: Development, local testing, cost-sensitive apps
""")

print("\n" + "="*60)
print("✓ See individual files for detailed demonstrations")
print("="*60)
