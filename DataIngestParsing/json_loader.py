# JSON File Loading and Processing

from langchain_community.document_loaders import JSONLoader
import json

print("Starting JSON Loading Pipeline...\n")

json_path = "../data/json/dummy.json"

print("="*60)
print("LOADING JSON FILE")
print("="*60)

try:
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".employees[]",
        text_content=False
    )
    
    docs = loader.load()
    print(f"\n✓ Successfully loaded {len(docs)} employees from dummy.json\n")
    
    for index, doc in enumerate(docs, start=1):
        employee_data = json.loads(doc.page_content)
        
        print(f"--- Employee {index} ---")
        print(f"Name: {employee_data.get('name')}")
        print(f"Department: {employee_data.get('department')}")
        print(f"Role: {employee_data.get('role')}")
        print(f"Email: {employee_data.get('email')}")
        print(f"Location: {employee_data.get('location')}")
        print(f"Skills: {employee_data.get('skills')}")
        print(f"Experience (years): {employee_data.get('experience_years')}")
        print(f"Salary: {employee_data.get('salary')}")
        print(f"Manager: {employee_data.get('manager')}")
        print(f"Projects: {len(employee_data.get('projects', []))} projects")
        print()
        
except Exception as e:
    print(f"✗ Error loading JSON: {e}")

print("✓ JSON loading completed!")
