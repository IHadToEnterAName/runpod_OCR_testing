"""
Visual Store Module
====================
Byaldi (ColQwen2) wrapper for visual document indexing and retrieval.
Replaces ChromaDB text-based vector store with visual embeddings.
"""

import os
import shutil
import base64
import io
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from PIL import Image

from config.settings import get_config

config = get_config()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PageResult:
    """A single page result from visual search."""
    page_number: int
    document_name: str
    score: float
    image_base64: str  # Base64-encoded PNG

@dataclass
class SearchResults:
    """Collection of search results."""
    results: List[PageResult]
    query: str
    total_pages_searched: int

# =============================================================================
# VISUAL STORE (Byaldi Wrapper)
# =============================================================================

class VisualStore:
    """
    Manages visual document indexes using Byaldi (ColQwen2).

    Byaldi handles:
    - PDF page screenshots (internal)
    - ColQwen2 visual embeddings
    - Similarity search
    - Index persistence to disk
    """

    def __init__(self):
        self._model = None
        self._index_base_path = config.byaldi.index_path
        self._active_indexes: Dict[str, Any] = {}

        os.makedirs(self._index_base_path, exist_ok=True)

    def _get_model(self):
        """Lazy-load the ColQwen2 model via Byaldi."""
        if self._model is None:
            from byaldi import RAGMultiModalModel
            print(f"Loading Byaldi model: {config.byaldi.model_name}")
            self._model = RAGMultiModalModel.from_pretrained(
                config.byaldi.model_name
            )
            print("Byaldi model loaded successfully")
        return self._model

    def create_index(self, index_name: str, file_path: str, file_name: str) -> int:
        """
        Create a new index from a document file.

        Args:
            index_name: Unique name for this index (session-based)
            file_path: Path to the PDF/image file
            file_name: Original filename for metadata

        Returns:
            Number of pages indexed
        """
        model = self._get_model()

        index_path = os.path.join(self._index_base_path, index_name)

        print(f"Indexing {file_name} into {index_name}...")
        model.index(
            input_path=file_path,
            index_name=index_name,
            store_collection_with_index=config.byaldi.store_collection_with_index,
            overwrite=True
        )

        # Get page count from the indexed data
        page_count = self._get_page_count(index_name)
        # For single images, Byaldi may not report count correctly
        if page_count == 0:
            page_count = 1

        # Track metadata for this index
        self._active_indexes[index_name] = {
            "documents": [file_name],
            "path": index_path,
            "total_pages": page_count
        }

        print(f"Indexed {page_count} pages from {file_name}")

        return page_count

    def add_to_index(self, index_name: str, file_path: str, file_name: str) -> int:
        """
        Add a document to an existing index.

        Args:
            index_name: Existing index name
            file_path: Path to the new document
            file_name: Original filename

        Returns:
            Number of new pages added
        """
        model = self._get_model()

        print(f"Adding {file_name} to index {index_name}...")
        model.add_to_index(
            input_item=file_path,
            store_collection_with_index=config.byaldi.store_collection_with_index
        )

        # Get updated page count
        prev_count = self._active_indexes.get(index_name, {}).get("total_pages", 0)
        page_count = self._get_page_count(index_name)
        if page_count <= prev_count:
            page_count = prev_count + 1  # At least 1 new page

        # Update metadata
        if index_name in self._active_indexes:
            self._active_indexes[index_name]["documents"].append(file_name)
            self._active_indexes[index_name]["total_pages"] = page_count

        print(f"Index {index_name} now has {page_count} pages")

        return page_count

    def search(self, index_name: str, query: str, top_k: int = None) -> SearchResults:
        """
        Search for relevant pages using visual similarity.

        Args:
            index_name: Index to search
            query: Natural language query
            top_k: Number of results (defaults to config.visual_rag.search_top_k)

        Returns:
            SearchResults with page images and scores
        """
        if top_k is None:
            top_k = config.visual_rag.search_top_k

        model = self._get_model()

        # Load the index if not already loaded
        self._ensure_index_loaded(index_name)

        results = model.search(query, k=top_k)

        page_results = []
        for result in results:
            # Get the page image as base64
            image_base64 = self._get_page_image_base64(result)

            # Determine document name
            doc_name = self._get_document_name(index_name, result)

            page_results.append(PageResult(
                page_number=result.page_num + 1,  # Byaldi uses 0-indexed pages
                document_name=doc_name,
                score=result.score,
                image_base64=image_base64
            ))

        return SearchResults(
            results=page_results,
            query=query,
            total_pages_searched=self._get_page_count(index_name)
        )

    def delete_index(self, index_name: str):
        """Delete an index and its files from disk."""
        index_path = os.path.join(self._index_base_path, index_name)

        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            print(f"Deleted index: {index_name}")

        # Also check for .byaldi directory
        byaldi_path = os.path.join(".byaldi", index_name)
        if os.path.exists(byaldi_path):
            shutil.rmtree(byaldi_path)

        self._active_indexes.pop(index_name, None)

    def get_stats(self, index_name: str) -> Dict[str, Any]:
        """Get statistics about an index."""
        meta = self._active_indexes.get(index_name, {})
        page_count = self._get_page_count(index_name)

        return {
            "index_name": index_name,
            "documents": meta.get("documents", []),
            "total_pages": page_count,
            "document_count": len(meta.get("documents", []))
        }

    def has_documents(self, index_name: str) -> bool:
        """Check if an index has any documents (in memory or on disk)."""
        if index_name in self._active_indexes:
            return self._get_page_count(index_name) > 0
        return self.index_exists_on_disk(index_name)

    def index_exists_on_disk(self, index_name: str) -> bool:
        """Check if a persisted index exists on disk."""
        index_path = os.path.join(self._index_base_path, index_name)
        if os.path.exists(index_path) and os.listdir(index_path):
            return True
        byaldi_path = os.path.join(".byaldi", index_name)
        if os.path.exists(byaldi_path) and os.listdir(byaldi_path):
            return True
        return False

    def load_existing_index(self, index_name: str) -> bool:
        """
        Load an existing index from disk into memory.

        Returns:
            True if index was loaded, False if not found.
        """
        if index_name in self._active_indexes:
            return True

        index_path = os.path.join(self._index_base_path, index_name)
        if not os.path.exists(index_path):
            return False

        try:
            model = self._get_model()
            model.load_index(index_name)
            self._active_indexes[index_name] = {
                "documents": [],
                "path": index_path
            }
            print(f"Loaded existing index: {index_name}")
            return True
        except Exception as e:
            print(f"Failed to load index {index_name}: {e}")
            return False

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _ensure_index_loaded(self, index_name: str):
        """Load an index from disk if not already in memory."""
        model = self._get_model()

        # Check if we need to load from disk
        index_path = os.path.join(self._index_base_path, index_name)
        if os.path.exists(index_path) and index_name not in self._active_indexes:
            model.load_index(index_name)
            self._active_indexes[index_name] = {
                "documents": [],
                "path": index_path
            }

    def _get_page_count(self, index_name: str) -> int:
        """Get the number of pages in an index."""
        try:
            model = self._get_model()

            # Try Byaldi/ColPali internal attributes
            if hasattr(model, 'model'):
                inner = model.model
                # ColPaliModel stores embeddings per page
                if hasattr(inner, 'indexed_embeddings') and inner.indexed_embeddings is not None:
                    return len(inner.indexed_embeddings)
                if hasattr(inner, 'collection') and inner.collection is not None:
                    return len(inner.collection)
                # doc_ids is a list with one entry per page
                if hasattr(inner, 'doc_ids') and inner.doc_ids is not None:
                    return len(inner.doc_ids)

            # Fallback: manually tracked count
            meta = self._active_indexes.get(index_name, {})
            return meta.get("total_pages", 0)
        except Exception:
            return 0

    def _get_page_image_base64(self, result) -> str:
        """Extract page image as base64 from a Byaldi search result."""
        try:
            # Byaldi stores images when store_collection_with_index=True
            if hasattr(result, 'base64'):
                return result.base64

            # Fallback: if result has PIL image
            if hasattr(result, 'image') and result.image is not None:
                return pil_to_base64(result.image)

            # Fallback: if result has metadata with image path
            if hasattr(result, 'metadata') and 'image_path' in result.metadata:
                img = Image.open(result.metadata['image_path'])
                return pil_to_base64(img)

            return ""
        except Exception as e:
            print(f"Warning: Could not extract page image: {e}")
            return ""

    def _get_document_name(self, index_name: str, result) -> str:
        """Get the document name for a search result."""
        if hasattr(result, 'doc_id'):
            meta = self._active_indexes.get(index_name, {})
            docs = meta.get("documents", [])
            doc_id = result.doc_id
            if doc_id < len(docs):
                return docs[doc_id]
        return "document"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# =============================================================================
# SINGLETON
# =============================================================================

_store_instance = None

def get_visual_store() -> VisualStore:
    """Get the global VisualStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = VisualStore()
    return _store_instance
