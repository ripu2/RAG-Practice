# Introduction to data ingestion

import os
from typing import List, Dict, Any
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
)

print("Starting data ingestion pipeline...")

# understand doc structure

doc = Document(
    page_content="Hello, world!",
    metadata={
        "source": "example.txt",
        "page": 1,
        "author": "Ripu",
        "date_created": "2024-01-01",
        "custom_field": "some information"
    }
)

print(doc)

print("Document content:")
print(f"content:{doc.page_content}")
print("Document metadata:")
print(f"Metadata:{doc.metadata}")

# Text files

os.makedirs("data/text", exist_ok=True)

sample_text_file = {
    "data/text/sample.txt": """PYTHON INTRODUCTION

1. What is Python?

Python is a high-level, interpreted programming language.
It is easy to read, easy to write, and beginner-friendly.
Python is widely used in web development, data science, AI, automation, and more.

2. Why Python?

- Simple and readable syntax
- Large community support
- Huge number of libraries
- Cross-platform (Windows, Mac, Linux)

3. Basic Syntax Example

Print statement:
print("Hello, World!")

Variables:
name = "John"
age = 25

4. Data Types

- int       -> 10
- float     -> 10.5
- string    -> "Hello"
- boolean   -> True / False
- list      -> [1, 2, 3]
- dictionary-> {"name": "John", "age": 25}

5. Conditional Statements

if age > 18:
    print("Adult")
else:
    print("Minor")

6. Loops

For loop:
for i in range(5):
    print(i)

While loop:
count = 0
while count < 5:
    print(count)
    count += 1

7. Functions

def greet(name):
    return "Hello " + name

8. Applications of Python

- Web Development (Django, Flask)
- Data Science (Pandas, NumPy)
- Machine Learning (TensorFlow, PyTorch)
- Automation & Scripting
- Game Development

Conclusion:

Python is powerful, simple, and one of the most in-demand programming languages in the world.
It is a great language for beginners as well as experienced developers. """
}

for filePath, content in sample_text_file.items():
    with open(filePath, "w", encoding="utf-8") as file:
        file.write(content)

print("Text files created successfully.")

# Text Loader Single File

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/text/sample.txt", encoding="utf-8")
docs = loader.load()

print(f"docs: {docs}")

# Directory Loader - Multiple Text File

# load all files from a directory

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    "data/text",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)
docs = loader.load()

for i, doc in enumerate(docs, start=1):
    print(f"doc:  {i}")
    print(f"\n Source: {doc.metadata['source']}")
    print(f"\n Content: {doc.page_content}")
    print("\n")

# Document Splitting

# Character Text Splitter

from langchain_text_splitters import CharacterTextSplitter
text = docs[0].page_content

character_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

chunks = character_splitter.split_text(text)

print("chunks Created")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print("\n")

# Recursive splitter, recc

from langchain_text_splitters import RecursiveCharacterTextSplitter
chunk_size = 200
chunk_overlap = chunk_size * 0.3

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[" "],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)

chunks = recursive_splitter.split_text(text)

print("Recursive Chunks Created")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print("\n")

# Loading PDF

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyMuPDFLoader,
    UnstructuredPDFLoader,
)

try:
    loader = PyPDFLoader("../data/pdf/attention.pdf")
    docs = loader.load()
    print(f"✓ Successfully loaded {len(docs)} pages from attention.pdf\n")
    print(f"Document metadata:")
    print(f"  - Creator: {docs[0].metadata.get('creator')}")
    print(f"  - Creation date: {docs[0].metadata.get('creationdate')}")
    print(f"  - Total pages: {docs[0].metadata.get('total_pages')}\n")

    print(f"First page preview:")
    print(f"{'-'*60}")
    print(f"{docs[0].page_content[:400]}...")
    print(f"{'-'*60}")
except Exception as e:
    print(f"Error loading PDF: {e}")

try:
    loader = PyMuPDFLoader("../data/pdf/attention.pdf")
    docs = loader.load()
    print(f"✓ Successfully loaded {len(docs)} pages from attention.pdf\n")
    print(f"Document metadata:")
    print(f"  - Creator: {docs[0].metadata.get('creator')}")
    print(f"  - Creation date: {docs[0].metadata.get('creationdate')}")
    print(f"  - Total pages: {docs[0].metadata.get('total_pages')}\n")
    print(f"First page preview:")
    print(f"{'-'*60}")
    print(f"{docs[0].page_content[:400]}...")
    print(f"{'-'*60}")
except Exception as e:
    print(f"Error loading PDF: {e}")
