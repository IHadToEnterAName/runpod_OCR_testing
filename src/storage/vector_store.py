"""
Vector Store Module
===================
ChromaDB operations with original logic preserved exactly.
"""

import chromadb
from chromadb.config import Settings
import uuid
import re
from typing import List, Dict, Optional
from config.settings import get_config

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()

# Initialize ChromaDB client (original logic)
chroma_client = chromadb.PersistentClient(
    path=config.database.chroma_persist_dir,
    settings=Settings(anonymized_telemetry=False, allow_reset=True)
)

print(f"✅ ChromaDB at {config.database.chroma_persist_dir}")

# =============================================================================
# RETRIEVAL FUNCTION (Original Logic)
# =============================================================================

def retrieve_chunks(
    collection: chromadb.Collection,
    query: str,
    query_embedding: List[float],
    top_k: int = None,
    page_filter: Optional[int] = None
) -> List[Dict]:
    """
    Retrieve chunks from ChromaDB.
    Original logic from your code preserved exactly.
    """
    if top_k is None:
        top_k = config.chunking.max_context_chunks
    
    try:
        # Check collection size
        count = collection.count()
        print(f"📊 Collection has {count} chunks")
        
        if count == 0:
            print("⚠️ Collection is empty!")
            return []
        
        # Check for page filter in query
        page_match = re.search(r'page\s*(\d+)', query.lower())
        if page_match:
            page_filter = int(page_match.group(1))
            print(f"📄 Page filter detected: {page_filter}")
        
        chunks = []
        
        # If page filter, try page-specific retrieval first
        if page_filter:
            try:
                page_results = collection.get(
                    where={"page_number": page_filter},
                    include=["documents", "metadatas"]
                )
                
                if page_results and page_results.get("documents"):
                    print(f"📄 Found {len(page_results['documents'])} chunks from page {page_filter}")
                    
                    for doc, meta in zip(
                        page_results["documents"], 
                        page_results["metadatas"]
                    ):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text")
                        })
                    
                    # If we have enough, return
                    if len(chunks) >= top_k // 2:
                        return chunks[:top_k]
                        
            except Exception as e:
                print(f"⚠️ Page filter failed: {e}")
        
        # Semantic search (original logic)
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if results and results.get("documents") and results["documents"][0]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ):
                    # Avoid duplicates
                    if not any(c["text"][:50] == doc[:50] for c in chunks):
                        chunks.append({
                            "text": doc,
                            "page": meta.get("page_number"),
                            "type": meta.get("chunk_type", "text"),
                            "distance": dist
                        })
                
                print(f"✅ Retrieved {len(chunks)} total chunks")
            else:
                print("⚠️ No results from semantic search")
                
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            import traceback
            traceback.print_exc()
        
        return chunks[:top_k]
        
    except Exception as e:
        print(f"❌ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return []

# =============================================================================
# COLLECTION MANAGEMENT
# =============================================================================

def create_collection(name: str) -> chromadb.Collection:
    """Create or get collection."""
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def delete_collection(name: str):
    """Delete collection."""
    try:
        chroma_client.delete_collection(name)
        print(f"🗑️ Deleted collection: {name}")
    except:
        pass

def add_chunks_to_collection(
    collection: chromadb.Collection,
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
    ids: List[str]
):
    """
    Add chunks to collection.
    Original batch logic preserved.
    """
    batch_size = 50
    
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        
        batch_docs = documents[i:batch_end]
        batch_embs = embeddings[i:batch_end]
        batch_metas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]
        
        collection.add(
            documents=batch_docs,
            embeddings=batch_embs,
            metadatas=batch_metas,
            ids=batch_ids
        )
        
        print(f"   Added {batch_end}/{len(documents)} chunks...")
