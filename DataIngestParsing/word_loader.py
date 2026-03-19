# Word Document Loading and Processing

from docx import Document as DocxDocument
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)

print("Starting Word Document Loading Pipeline...\n")

doc_path = "../data/doc/data-processing-agreement.docx"

# Method 1: Docx2txtLoader
print("="*60)
print("LOADING WORD DOCUMENT WITH Docx2txtLoader")
print("="*60)

try:
    loader = Docx2txtLoader(doc_path)
    docs = loader.load()
    print(f"\n✓ Successfully loaded {len(docs)} document(s) from data-processing-agreement.docx\n")
    
    print(f"Document metadata:")
    print(f"  - Source: {docs[0].metadata.get('source')}")
    
    print(f"\nDocument content preview:")
    print(f"{'-'*60}")
    print(f"{docs[0].page_content[:400]}...")
    print(f"{'-'*60}")
    
except Exception as e:
    print(f"✗ Error loading Word document: {e}")

# Method 2: UnstructuredWordDocumentLoader
print("\n" + "="*60)
print("LOADING WORD DOCUMENT WITH UnstructuredWordDocumentLoader")
print("="*60)

try:
    loader = UnstructuredWordDocumentLoader(doc_path, mode="elements")
    docs = loader.load()
    
    print(f"\n✓ Successfully loaded {len(docs)} elements from data-processing-agreement.docx\n")

    for pageIndex, doc in enumerate(docs[:5], start=1):
        print(f"--- Element {pageIndex} ---")
        print(f"Category: {doc.metadata.get('category', 'unknown')}")
        print(f"Content: {doc.page_content}")
        print()
        
    if len(docs) > 5:
        print(f"... (showing 5 of {len(docs)} elements)")
        
except Exception as e:
    print(f"✗ Error loading Word document: {e}")

print("\n✓ Word document loading completed!")
