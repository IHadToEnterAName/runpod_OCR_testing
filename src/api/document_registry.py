"""
Document Registry
==================
JSON-based metadata store for tracking uploaded documents.
Maps documentId to document info (name, page ranges, file path).
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from config.settings import get_config

config = get_config()


@dataclass
class DocumentRecord:
    """Metadata for a single uploaded document."""
    document_id: str
    document_name: str
    file_path: str        # Path to the stored original file
    chunk_count: int      # Number of chunks (pages)
    byaldi_doc_id: int    # Byaldi's internal doc_id in the shared index
    file_type: str        # pdf, docx, txt, json, image


class DocumentRegistry:
    """
    Persistent registry of uploaded documents.
    Stored as a JSON file alongside the Byaldi index.
    """

    def __init__(self, registry_dir: str = None):
        if registry_dir is None:
            registry_dir = os.path.join(config.byaldi.index_path, "api_registry")
        self._registry_dir = registry_dir
        self._registry_path = os.path.join(registry_dir, "documents.json")
        self._documents: Dict[str, DocumentRecord] = {}
        os.makedirs(registry_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Load registry from disk."""
        if os.path.exists(self._registry_path):
            try:
                with open(self._registry_path, 'r') as f:
                    data = json.load(f)
                self._documents = {
                    doc_id: DocumentRecord(**record)
                    for doc_id, record in data.items()
                }
            except (json.JSONDecodeError, IOError, TypeError):
                self._documents = {}

    def _save(self):
        """Persist registry to disk."""
        data = {
            doc_id: asdict(record)
            for doc_id, record in self._documents.items()
        }
        with open(self._registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add(self, record: DocumentRecord):
        """Register a new document."""
        self._documents[record.document_id] = record
        self._save()

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        """Get a document record by ID."""
        return self._documents.get(document_id)

    def remove(self, document_id: str) -> Optional[DocumentRecord]:
        """Remove a document record. Returns the removed record or None."""
        record = self._documents.pop(document_id, None)
        if record is not None:
            self._save()
        return record

    def list_all(self) -> List[DocumentRecord]:
        """List all registered documents."""
        return list(self._documents.values())

    def get_all_except(self, document_id: str) -> List[DocumentRecord]:
        """Get all documents except the specified one (used for index rebuild)."""
        return [r for r in self._documents.values() if r.document_id != document_id]

    def get_by_byaldi_doc_id(self, byaldi_doc_id: int) -> Optional[DocumentRecord]:
        """Find document record by Byaldi's internal doc_id."""
        for record in self._documents.values():
            if record.byaldi_doc_id == byaldi_doc_id:
                return record
        return None

    def next_byaldi_doc_id(self) -> int:
        """Get the next available Byaldi doc_id."""
        if not self._documents:
            return 0
        return max(r.byaldi_doc_id for r in self._documents.values()) + 1

    def is_empty(self) -> bool:
        return len(self._documents) == 0

    def count(self) -> int:
        return len(self._documents)


# =============================================================================
# SINGLETON
# =============================================================================

_registry_instance = None


def get_document_registry() -> DocumentRegistry:
    """Get the global DocumentRegistry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = DocumentRegistry()
    return _registry_instance
