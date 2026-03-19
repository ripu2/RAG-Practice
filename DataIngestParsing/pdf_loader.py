# PDF Loading and Processing

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import langchain_core.documents as Document

print("Starting PDF Loading Pipeline...\n")

pdf_path = "../data/pdf/attention.pdf"

# PyPDFLoader
print("="*60)
print("LOADING PDF WITH PyPDFLoader")
print("="*60)

try:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"\n✓ Successfully loaded {len(docs)} pages from attention.pdf\n")
    print(f"Document metadata:")
    print(f"  - Creator: {docs[0].metadata.get('creator')}")
    print(f"  - Creation date: {docs[0].metadata.get('creationdate')}")
    print(f"  - Total pages: {docs[0].metadata.get('total_pages')}\n")

    print(f"First page preview:")
    print(f"{'-'*60}")
    print(f"{docs[0].page_content[:400]}...")
    print(f"{'-'*60}")
except Exception as e:
    print(f"✗ Error loading PDF: {e}")

# PyMuPDFLoader
print("\n" + "="*60)
print("LOADING PDF WITH PyMuPDFLoader")
print("="*60)

try:
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"\n✓ Successfully loaded {len(docs)} pages from attention.pdf\n")
    print(f"Document metadata:")
    print(f"  - Creator: {docs[0].metadata.get('creator')}")
    print(f"  - Creation date: {docs[0].metadata.get('creationdate')}")
    print(f"  - Total pages: {docs[0].metadata.get('total_pages')}\n")
    print(f"First page preview:")
    print(f"{'-'*60}")
    print(f"{docs[0].page_content[:400]}...")
    print(f"{'-'*60}")
except Exception as e:
    print(f"✗ Error loading PDF: {e}")

# Smart PDF Loader Class
print("\n" + "="*60)
print("SMART PDF LOADER WITH CHUNKING")
print("="*60)


class SmartPDFLoader:
    """Advanced PDF loader that handles with error handling and chunking"""

    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[" "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """Process a PDF file and return a list of documents"""

        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            parsed_chunks = []

            for page_num, page in enumerate(pages):
                clean_text = self.clean_text(page.page_content)

                if len(clean_text.strip()) < 50:
                    continue

                chunks = self.text_splitter.create_documents(
                    texts=[clean_text],
                    metadatas=[{
                        **page.metadata,
                        "page": page_num + 1,
                        "total_pages": len(pages),
                        "chunk_method": "smart_pdf_processor",
                        "char_count": len(clean_text),
                    }]
                )

                parsed_chunks.extend(chunks)

            return parsed_chunks

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """Clean the text by removing extra whitespace and newlines"""
        text = " ".join(text.split())
        text = text.replace("fi", "fi")
        text = text.replace("fl", "fl")

        return text


# Process PDF with Smart Loader
preprocessor = SmartPDFLoader()
smart_chunks = preprocessor.process_pdf(pdf_path)

print(f"\n✓ Total chunks created: {len(smart_chunks)}")

if smart_chunks:
    print("\nFirst chunk metadata:")
    for key, value in smart_chunks[0].metadata.items():
        print(f"  {key}: {value}")

    print(f"\nFirst chunk content preview:")
    print(f"{smart_chunks[0].page_content[:200]}...")

print("\n✓ PDF loading completed!")
