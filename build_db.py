import os
import time
from dotenv import load_dotenv
from src.app import load_documents
from src.vectordb import VectorDB


load_dotenv()

def build_vector_database():
    """
    Loads documents, chunks them, generates embeddings, builds BM25 index,
    and saves everything to a persistent ChromaDB collection.
    """
    print("--- Starting Database Build Process ---")
    start_time = time.time()

    # Load Documents
    print("\n[Step 1/3] Loading PDF documents...")
    documents = load_documents()
    if not documents:
        print("❌ No documents found in 'data' directory. Aborting.")
        return
    print(f"✅ Loaded {len(documents)} document pages.")

    # Initialize VectorDB
    print("\n[Step 2/3] Initializing Vector Database and Embedding Model...")
    try:
        vector_db = VectorDB() 
    except Exception as e:
        print(f"❌ Error initializing VectorDB: {e}")
        return
    print("✅ VectorDB initialized.")

    # Process Documents: Chunking, Embedding, Indexing, Saving
    print("\n[Step 3/3] Processing documents (Chunking, Embedding, Indexing, Saving)...")
    print("   This may take a significant amount of time, depending on the number of documents and CPU speed.")
    try:
        
        vector_db.add_documents(documents)
    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        return
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n--- Database Build Process Complete ---")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    
    try:
        stats = vector_db.get_collection_stats()
        print("\nFinal Database Statistics:")
        print(f"   • Total chunks indexed: {stats['total_chunks']}")
        print(f"   • Collection name: {stats['collection_name']}")
    except Exception as e:
         print(f"⚠️ Could not retrieve final stats: {e}")


if __name__ == "__main__":
    build_vector_database()
