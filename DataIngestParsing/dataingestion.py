# Complete Data Ingestion Pipeline
# This file combines all data loading and processing functionality

import os
from typing import List, Dict, Any
import pandas as pd
import json

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader,
)
from docx import Document as DocxDocument

print("="*60)
print("DATA INGESTION PIPELINE")
print("="*60)

# For detailed examples, run the individual files:
# - text_loader.py
# - document_splitter.py
# - pdf_loader.py
# - word_loader.py
# - json_loader.py

print("\n✓ All loaders imported successfully!")
print("\nAvailable loaders:")
print("  - TextLoader (single file)")
print("  - DirectoryLoader (multiple files)")
print("  - PyPDFLoader (PDF files)")
print("  - PyMuPDFLoader (PDF files - better)")
print("  - Docx2txtLoader (Word documents)")
print("  - UnstructuredWordDocumentLoader (Word documents - advanced)")
print("  - JSONLoader (JSON files)")
print("\nAvailable text splitters:")
print("  - CharacterTextSplitter")
print("  - RecursiveCharacterTextSplitter")
print("  - TokenTextSplitter")
