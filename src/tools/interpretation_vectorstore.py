"""
Interpretation Vector Store for FAISS-based semantic search.

Provides:
- Embedding final_interpretations via gemini-embedding-001
- Building and saving FAISS indexes
- Loading cached indexes to avoid re-embedding
- Semantic search with hypothesis-derived queries
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Default embedding model
EMBEDDING_MODEL = "models/gemini-embedding-001"


def _roi_key_to_str(roi_key) -> str:
    """Convert roi_key (tuple or str) to a stable string representation."""
    if isinstance(roi_key, tuple):
        return f"({roi_key[0]}, {roi_key[1]})"
    return str(roi_key)


def _compute_content_hash(interpretations: Dict) -> str:
    """Compute a SHA-256 hash over interpretation content for cache invalidation."""
    content = json.dumps(
        {str(k): v for k, v in sorted(interpretations.items())},
        sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def merge_interpretations(
    live_interpretations: Dict,
    mock_interpretations: Optional[Dict] = None
) -> Dict[str, str]:
    """
    Merge live pipeline results with mock data.

    Live results take precedence (overwrite) mock data for same keys.

    Args:
        live_interpretations: Dict from pipeline (roi_key -> str)
        mock_interpretations: Optional Dict from pickle (roi_key -> str)

    Returns:
        Merged dict with all interpretations, keys normalized to strings
    """
    merged = {}

    # Add mock data first
    if mock_interpretations is not None:
        for key, val in mock_interpretations.items():
            merged[_roi_key_to_str(key)] = val

    # Live data overwrites
    for key, val in live_interpretations.items():
        merged[_roi_key_to_str(key)] = val

    return merged


class InterpretationVectorStore:
    """
    FAISS vector store for ROI final_interpretations.

    Manages embedding, indexing, caching, and searching.
    """

    def __init__(
        self,
        index_dir: Path = Path("./outputs/faiss_index"),
        embedding_model_name: str = EMBEDDING_MODEL,
    ):
        self.index_dir = Path(index_dir)
        self.embedding_model_name = embedding_model_name
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model_name)
        self.vectorstore: Optional[FAISS] = None

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def index_exists(self) -> bool:
        """Check if a cached FAISS index exists on disk."""
        return (
            (self.index_dir / "index.faiss").exists()
            and (self.index_dir / "index.pkl").exists()
            and self.manifest_path.exists()
        )

    def _read_manifest(self) -> Optional[Dict]:
        """Read the cache manifest from disk."""
        if not self.manifest_path.exists():
            return None
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _write_manifest(self, roi_keys: List[str], content_hash: str) -> None:
        """Write cache manifest to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": datetime.now().isoformat(),
            "embedding_model": self.embedding_model_name,
            "num_documents": len(roi_keys),
            "content_hash": content_hash,
            "roi_keys": roi_keys
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def is_cache_valid(self, interpretations: Dict) -> bool:
        """
        Check if the cached index matches the current data.

        Uses content hash comparison for cache invalidation.
        """
        if not self.index_exists:
            return False
        manifest = self._read_manifest()
        if manifest is None:
            return False
        # Also check embedding model hasn't changed
        if manifest.get("embedding_model") != self.embedding_model_name:
            return False
        current_hash = _compute_content_hash(interpretations)
        return manifest.get("content_hash") == current_hash

    def build_index(
        self,
        interpretations: Dict,
        force_rebuild: bool = False
    ) -> FAISS:
        """
        Build FAISS index from interpretations.

        If a valid cached index exists and force_rebuild is False,
        loads from disk instead of re-embedding.

        Args:
            interpretations: Dict mapping roi_key (str or tuple) -> final_interpretation text
            force_rebuild: If True, ignore cache and rebuild

        Returns:
            FAISS vectorstore instance
        """
        # Normalize keys to strings
        normalized = {_roi_key_to_str(k): v for k, v in interpretations.items()}

        # Check cache
        if not force_rebuild and self.is_cache_valid(normalized):
            print(f"  Loading cached FAISS index from {self.index_dir}")
            return self.load_index()

        print(f"  Building new FAISS index with {len(normalized)} documents...")

        # Create LangChain Documents with metadata
        documents = []
        for roi_key_str, text in normalized.items():
            doc = Document(
                page_content=text,
                metadata={
                    "roi_key": roi_key_str,
                    "source": "final_interpretation"
                }
            )
            documents.append(doc)

        # Build FAISS index from documents (embeds all documents)
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        # Save to disk
        self.save_index(normalized)

        return self.vectorstore

    def save_index(self, interpretations: Dict[str, str]) -> None:
        """Save FAISS index and manifest to disk."""
        if self.vectorstore is None:
            raise ValueError("No vectorstore to save. Call build_index first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))

        # Write manifest
        content_hash = _compute_content_hash(interpretations)
        roi_keys = list(interpretations.keys())
        self._write_manifest(roi_keys, content_hash)

        print(f"  Saved FAISS index to {self.index_dir}")
        print(f"    Documents: {len(roi_keys)}")
        print(f"    Content hash: {content_hash}")

    def load_index(self) -> FAISS:
        """
        Load cached FAISS index from disk.

        Returns:
            FAISS vectorstore instance
        """
        if not self.index_exists:
            raise FileNotFoundError(
                f"No FAISS index found at {self.index_dir}. "
                "Call build_index first."
            )

        self.vectorstore = FAISS.load_local(
            folder_path=str(self.index_dir),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for ROIs matching a semantic query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of dicts with keys: roi_key, text, score
        """
        if self.vectorstore is None:
            self.load_index()

        # Perform similarity search with scores (L2 distance, lower = more similar)
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k
        )

        # Format results
        search_results = []
        for doc, score in results:
            search_results.append({
                "roi_key": doc.metadata["roi_key"],
                "text": doc.page_content,
                "score": float(score),
                "source": doc.metadata.get("source", "unknown")
            })

        return search_results

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        manifest = self._read_manifest()
        if manifest is None:
            return {"status": "no_index"}

        return {
            "status": "ready",
            "index_dir": str(self.index_dir),
            "num_documents": manifest.get("num_documents", 0),
            "embedding_model": manifest.get("embedding_model", "unknown"),
            "created_at": manifest.get("created_at", "unknown"),
            "content_hash": manifest.get("content_hash", "unknown")
        }
