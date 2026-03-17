# Text File Loading and Processing

import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader

print("Starting Text File Loading Pipeline...")

# Understand document structure
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
print("\nDocument content:")
print(f"content: {doc.page_content}")
print("\nDocument metadata:")
print(f"Metadata: {doc.metadata}")

# Create sample text files
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

print("\n✓ Text files created successfully.")

# Load single text file
print("\n" + "="*60)
print("LOADING SINGLE TEXT FILE")
print("="*60)

loader = TextLoader("data/text/sample.txt", encoding="utf-8")
docs = loader.load()

print(f"\nLoaded {len(docs)} document(s)")
print(f"First document preview: {docs[0].page_content[:100]}...")

# Load all text files from directory
print("\n" + "="*60)
print("LOADING MULTIPLE TEXT FILES FROM DIRECTORY")
print("="*60)

loader = DirectoryLoader(
    "data/text",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True
)
docs = loader.load()

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Document {i} ---")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content preview: {doc.page_content[:150]}...")

print("\n✓ Text loading completed!")
