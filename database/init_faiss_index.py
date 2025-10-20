#!/usr/bin/env python3
"""
Initialize FAISS vector index for ADRIAN memory system.
Creates the index structure and saves it to disk.
"""
import os
import sys
import faiss
import numpy as np
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.config import get_settings

def initialize_faiss_index():
    """
    Initialize FAISS index for semantic memory storage.
    Uses IndexFlatL2 for exact search with cosine similarity.
    """
    settings = get_settings()
    
    # Create index directory if it doesn't exist
    index_dir = Path(settings.faiss_index_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Index file paths
    index_file = index_dir / "index.faiss"
    metadata_file = index_dir / "metadata.json"
    
    print(f"Initializing FAISS index...")
    print(f"Dimension: {settings.embedding_dimension}")
    print(f"Index path: {index_file}")
    
    # Create FAISS index
    # IndexFlatL2 uses L2 distance (Euclidean)
    # For cosine similarity, vectors should be normalized before adding
    index = faiss.IndexFlatL2(settings.embedding_dimension)
    
    # Optional: Add index wrapping for ID mapping
    # This allows us to store custom IDs with vectors
    index_with_ids = faiss.IndexIDMap(index)
    
    # Save the empty index to disk
    faiss.write_index(index_with_ids, str(index_file))
    
    # Create metadata file
    metadata = {
        "dimension": settings.embedding_dimension,
        "index_type": "IndexIDMap(IndexFlatL2)",
        "total_vectors": 0,
        "created_at": str(np.datetime64('now')),
        "version": "1.0"
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ FAISS index created and saved")
    print(f"✓ Metadata file created")
    print(f"\nIndex info:")
    print(f"  - Vectors: {index_with_ids.ntotal}")
    print(f"  - Dimension: {settings.embedding_dimension}")
    print(f"  - Index type: IndexIDMap(IndexFlatL2)")
    
    return True


def verify_index():
    """Verify the created index can be loaded."""
    settings = get_settings()
    index_file = Path(settings.faiss_index_path) / "index.faiss"
    
    print(f"\nVerifying index...")
    
    try:
        # Load the index
        index = faiss.read_index(str(index_file))
        print(f"✓ Index loaded successfully")
        print(f"  - Total vectors: {index.ntotal}")
        print(f"  - Dimension: {index.d}")
        
        # Test adding a dummy vector
        test_vector = np.random.rand(1, settings.embedding_dimension).astype('float32')
        index.add_with_ids(test_vector, np.array([1]))
        
        print(f"✓ Test vector added successfully")
        print(f"  - Total vectors after test: {index.ntotal}")
        
        # Test search
        distances, indices = index.search(test_vector, k=1)
        print(f"✓ Search test successful")
        print(f"  - Found vector ID: {indices[0][0]}")
        print(f"  - Distance: {distances[0][0]:.6f}")
        
        # Don't save the test vector
        print(f"\n✓ Verification complete! Index is ready to use.")
        
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("ADRIAN FAISS Index Initialization")
    print("=" * 70)
    print()
    
    try:
        # Initialize index
        if initialize_faiss_index():
            print("\n" + "=" * 70)
            
            # Verify it works
            if verify_index():
                print("\n" + "=" * 70)
                print("SUCCESS: FAISS index is ready!")
                print("=" * 70)
                sys.exit(0)
            else:
                print("\nWARNING: Index created but verification failed")
                sys.exit(1)
        else:
            print("\nERROR: Failed to initialize index")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

