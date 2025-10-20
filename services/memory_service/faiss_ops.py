"""
FAISS vector operations for semantic memory search.
Handles embedding generation, indexing, and similarity search.
"""
import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from shared.config import get_settings
from shared.logging_config import get_logger

logger = get_logger("memory-service.faiss")
settings = get_settings()

# Global instances
_index: Optional[faiss.IndexIDMap] = None
_model: Optional[Any] = None  # Can be SentenceTransformer or False (if failed to load)
_next_id: int = 0


def get_embedding_model():
    """
    Get or load the sentence transformer model.
    Returns None if model cannot be loaded (PyTorch compatibility issue).
    """
    global _model
    if _model is None:
        try:
            logger.info("Loading sentence transformer model...")
            # Lazy import to avoid PyTorch loading at module import time
            from sentence_transformers import SentenceTransformer
            # Use all-MiniLM-L6-v2: 384 dimensions, fast, good quality
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            logger.warning("PyTorch/sentence-transformers not compatible with Python 3.13")
            logger.info("Using fallback hash-based embeddings")
            # Set to False to indicate model loading failed
            _model = False
    
    # Return None if model failed to load
    return _model if _model is not False else None


def load_index() -> faiss.IndexIDMap:
    """Load FAISS index from disk."""
    global _index, _next_id
    
    index_path = Path(settings.faiss_index_path) / "index.faiss"
    metadata_path = Path(settings.faiss_index_path) / "metadata.json"
    
    if not index_path.exists():
        logger.warning(f"Index file not found: {index_path}")
        logger.info("Creating new index...")
        
        # Create new index
        base_index = faiss.IndexFlatL2(settings.embedding_dimension)
        _index = faiss.IndexIDMap(base_index)
        _next_id = 0
        
        # Save it
        save_index()
        return _index
    
    # Load existing index
    _index = faiss.read_index(str(index_path))
    
    # Load metadata to get next ID
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            _next_id = metadata.get("total_vectors", 0)
    else:
        _next_id = _index.ntotal
    
    logger.info(f"Loaded FAISS index: {_index.ntotal} vectors")
    return _index


def save_index():
    """Save FAISS index to disk."""
    global _index, _next_id
    
    if _index is None:
        logger.warning("No index to save")
        return
    
    index_path = Path(settings.faiss_index_path) / "index.faiss"
    metadata_path = Path(settings.faiss_index_path) / "metadata.json"
    
    # Ensure directory exists
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save index
    faiss.write_index(_index, str(index_path))
    
    # Save metadata
    metadata = {
        "dimension": settings.embedding_dimension,
        "index_type": "IndexIDMap(IndexFlatL2)",
        "total_vectors": _index.ntotal,
        "updated_at": str(np.datetime64('now')),
        "version": "1.0"
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved FAISS index: {_index.ntotal} vectors")


async def generate_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector for text using sentence transformer.
    Returns normalized float32 numpy array.
    Falls back to simple hash-based embedding if transformer fails.
    """
    model = get_embedding_model()
    
    if model is not None:
        # Use real sentence transformer model
        try:
            embedding = model.encode(text, convert_to_numpy=True)
            
            # Ensure float32 type
            embedding = embedding.astype('float32')
            
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Fall through to fallback
    
    # Fallback: Create a simple hash-based embedding
    # This is temporary until PyTorch Python 3.13 compatibility is resolved
    import hashlib
    text_hash = hashlib.sha256(text.encode()).digest()
    
    # Convert hash to float vector of correct dimension
    seed = int.from_bytes(text_hash[:4], byteorder='big')
    np.random.seed(seed)
    embedding = np.random.rand(settings.embedding_dimension).astype('float32')
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding


async def add_to_index(text: str, custom_id: Optional[int] = None) -> int:
    """
    Add text embedding to FAISS index.
    Returns the assigned index ID.
    """
    global _index, _next_id
    
    if _index is None:
        load_index()
    
    # Generate embedding
    embedding = await generate_embedding(text)
    embedding = embedding.reshape(1, -1)  # Shape: (1, dimension)
    
    # Assign ID
    if custom_id is not None:
        vector_id = custom_id
    else:
        vector_id = _next_id
        _next_id += 1
    
    # Add to index
    _index.add_with_ids(embedding, np.array([vector_id], dtype=np.int64))
    
    logger.info(f"Added vector to index: ID={vector_id}")
    
    # Save index (in production, do this periodically, not every add)
    save_index()
    
    return vector_id


async def search_index(query_text: str, k: int = 5) -> List[Tuple[int, float]]:
    """
    Search FAISS index for similar embeddings.
    Returns list of (index_id, distance) tuples.
    """
    global _index
    
    if _index is None:
        load_index()
    
    if _index.ntotal == 0:
        logger.info("Index is empty, returning no results")
        return []
    
    # Generate query embedding
    query_embedding = await generate_embedding(query_text)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search
    # k = min(k, _index.ntotal)  # Don't search for more than we have
    distances, indices = _index.search(query_embedding, k)
    
    # Convert to list of tuples (excluding -1 which means no result)
    results = [
        (int(indices[0][i]), float(distances[0][i]))
        for i in range(len(indices[0]))
        if indices[0][i] != -1
    ]
    
    logger.info(f"Search returned {len(results)} results")
    return results


async def remove_from_index(vector_id: int):
    """
    Remove a vector from the index by ID.
    Note: FAISS IndexIDMap doesn't support removal directly.
    This would require rebuilding the index without the specified ID.
    """
    logger.warning("Vector removal requires index rebuild - not yet implemented")
    # TODO: Implement index rebuild without specific IDs
    pass


def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the FAISS index."""
    global _index
    
    if _index is None:
        return {
            "loaded": False,
            "total_vectors": 0
        }
    
    return {
        "loaded": True,
        "total_vectors": _index.ntotal,
        "dimension": _index.d,
        "index_type": type(_index).__name__
    }

