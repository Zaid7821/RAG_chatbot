print("Starting imports...")
try:
    import streamlit
    print("Imported streamlit")
    import chromadb
    print("Imported chromadb")
    from groq import Groq
    print("Imported Groq")
    from sentence_transformers import SentenceTransformer
    print("Imported SentenceTransformer")
except Exception as e:
    print(f"Error importing: {e}")
print("Imports done.")
