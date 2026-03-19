# Document Splitting and Chunking

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import TextLoader

print("Starting Document Splitting Pipeline...\n")

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

print(f"\n✓ chunks Created")
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print()

# Recursive Character Text Splitter
print("="*60)
print("RECURSIVE CHARACTER TEXT SPLITTER")
print("="*60)

chunk_size = 200
chunk_overlap = chunk_size * 0.3

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[" "],
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_overlap),
    length_function=len
)

chunks = recursive_splitter.split_text(text)

print(f"\n✓ Recursive Chunks Created")
print(f"Total chunks: {len(chunks)}\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print()

print("✓ Document splitting completed!")
