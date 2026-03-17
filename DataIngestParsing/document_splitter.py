# Document Splitting and Chunking

import os
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader

print("Starting Document Splitting Pipeline...")

# Load a sample document
loader = TextLoader("data/text/sample.txt", encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

print(f"✓ Loaded document with {len(text)} characters\n")

# Character Text Splitter
print("="*60)
print("CHARACTER TEXT SPLITTER")
print("="*60)

character_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

chunks = character_splitter.split_text(text)

print(f"\n✓ Created {len(chunks)} chunks using Character Splitter")
print("\nSample chunks:")

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)

print(f"\n... (showing 3 of {len(chunks)} chunks)")

# Recursive Character Text Splitter
print("\n" + "="*60)
print("RECURSIVE CHARACTER TEXT SPLITTER")
print("="*60)

chunk_size = 200
chunk_overlap = int(chunk_size * 0.3)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[" "],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)

chunks = recursive_splitter.split_text(text)

print(f"\n✓ Created {len(chunks)} chunks using Recursive Splitter")
print(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
print("\nSample chunks:")

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

print(f"\n... (showing 3 of {len(chunks)} chunks)")

# Advanced Recursive Splitter with multiple separators
print("\n" + "="*60)
print("ADVANCED RECURSIVE SPLITTER (Multiple Separators)")
print("="*60)

advanced_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)

chunks = advanced_splitter.split_text(text)

print(f"\n✓ Created {len(chunks)} chunks using Advanced Recursive Splitter")
print("Separators: newline-newline, newline, period-space, space, character")
print("\nSample chunks:")

for i, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)

print(f"\n... (showing 3 of {len(chunks)} chunks)")
print("\n✓ Document splitting completed!")
