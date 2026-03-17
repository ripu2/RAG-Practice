# PDF Loading and Processing

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
)

print("Starting PDF Loading Pipeline...")

pdf_path = "../data/pdf/attention.pdf"

# PyPDFLoader
print("\n" + "="*60)
print("LOADING PDF WITH PyPDFLoader")
print("="*60)

try:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"\n✓ Successfully loaded {len(docs)} pages from attention.pdf")
    
    print("\nDocument metadata:")
    print(f"  - Creator: {docs[0].metadata.get('creator')}")
    print(f"  - Creation date: {docs[0].metadata.get('creationdate')}")
    print(f"  - Total pages: {docs[0].metadata.get('total_pages')}")
    
    print("\nFirst page preview:")
    print("-" * 60)
    print(docs[0].page_content[:400] + "...")
    print("-" * 60)
    
    print(f"\nPage statistics:")
    print(f"  - Average page length: {sum(len(doc.page_content) for doc in docs) // len(docs)} chars")
    print(f"  - Shortest page: {min(len(doc.page_content) for doc in docs)} chars")
    print(f"  - Longest page: {max(len(doc.page_content) for doc in docs)} chars")
    
except Exception as e:
    print(f"✗ Error loading PDF with PyPDFLoader: {e}")

# PyMuPDFLoader
print("\n" + "="*60)
print("LOADING PDF WITH PyMuPDFLoader")
print("="*60)

try:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"\n✓ Successfully loaded {len(docs)} pages from attention.pdf")
    
    print("\nDocument metadata:")
    print(f"  - Creator: {docs[0].metadata.get('creator')}")
    print(f"  - Creation date: {docs[0].metadata.get('creationdate')}")
    print(f"  - Total pages: {docs[0].metadata.get('total_pages')}")
    
    print("\nFirst page preview:")
    print("-" * 60)
    print(docs[0].page_content[:400] + "...")
    print("-" * 60)
    
    print(f"\nPage statistics:")
    print(f"  - Average page length: {sum(len(doc.page_content) for doc in docs) // len(docs)} chars")
    print(f"  - Shortest page: {min(len(doc.page_content) for doc in docs)} chars")
    print(f"  - Longest page: {max(len(doc.page_content) for doc in docs)} chars")
    
    # Show sample pages
    print(f"\nSample page metadata:")
    for i in [0, len(docs)//2, -1]:
        page_num = docs[i].metadata.get('page', i)
        print(f"  - Page {page_num}: {len(docs[i].page_content)} chars")
    
except Exception as e:
    print(f"✗ Error loading PDF with PyMuPDFLoader: {e}")

print("\n✓ PDF loading completed!")

# Comparison summary
print("\n" + "="*60)
print("LOADER COMPARISON")
print("="*60)
print("""
PyPDFLoader:
  - Pure Python implementation
  - Good for simple PDFs
  - May have issues with complex layouts
  
PyMuPDFLoader:
  - Uses PyMuPDF (fitz) library
  - Better text extraction
  - Handles complex PDFs better
  - Faster performance
  
UnstructuredPDFLoader:
  - Most advanced extraction
  - Handles tables, images, layouts
  - Requires additional dependencies
""")
