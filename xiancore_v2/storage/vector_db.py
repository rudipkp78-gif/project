"""
FAISS-based Vector Database for Billion-Scale Fact Storage
Replaces simple LSH with production-grade vector search
Supports: HNSW, IVF-PQ, GPU acceleration, incremental indexing
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


class FAISSVectorStore:
    """
    Production-grade vector storage using FAISS
    Supports billion-scale similarity search with multiple index types
    """
    
    def __init__(self, dimension: int = 768, index_type: str = "HNSW",
                 metric: str = "cosine", gpu_id: int = -1):
        """
        dimension: Embedding dimension
        index_type: "HNSW" (fast), "IVF-PQ" (memory efficient), "Flat" (exact)
        metric: "cosine" or "l2"
        gpu_id: GPU device ID (-1 for CPU)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.gpu_id = gpu_id
        
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        self.index = None
        self.id_map = {}  # Maps internal FAISS IDs to external IDs
        self.metadata = {}  # Stores metadata for each vector
        self.total_vectors = 0
        
        self._initialize_index()
        
        if gpu_id >= 0:
            self._move_to_gpu(gpu_id)
    
    def _initialize_index(self):
        """Initialize FAISS index based on type"""
        if self.metric == "cosine":
            # Normalize vectors for cosine similarity
            self.normalize = True
        else:
            self.normalize = False
        
        if self.index_type == "HNSW":
            # HNSW: Fast approximate nearest neighbor search
            M = 32  # Number of connections per layer
            efConstruction = 200  # Size of dynamic candidate list
            self.index = self.faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = efConstruction
            
        elif self.index_type == "IVF-PQ":
            # IVF-PQ: Memory efficient for large datasets
            nlist = 1024  # Number of clusters
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            self.index = self.faiss.IndexIVFPQ(quantizer, self.dimension, nlist, 8, 8)
            # Note: Requires training before adding vectors
            
        elif self.index_type == "Flat":
            # Exact search (slow but accurate)
            if self.metric == "cosine":
                self.index = self.faiss.IndexFlatIP(self.dimension)
            else:
                self.index = self.faiss.IndexFlatL2(self.dimension)
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _move_to_gpu(self, gpu_id: int):
        """Move index to GPU for acceleration"""
        try:
            res = self.faiss.StandardGpuResources()
            self.index = self.faiss.index_cpu_to_gpu(res, gpu_id, self.index)
            print(f"Index moved to GPU {gpu_id}")
        except Exception as e:
            print(f"Failed to move to GPU: {e}. Continuing on CPU.")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str], 
                    metadata: Optional[List[Dict]] = None) -> int:
        """
        Add vectors to the index
        
        Args:
            vectors: np.ndarray of shape (num_vectors, dimension)
            ids: List of external IDs (strings)
            metadata: Optional list of metadata dicts
            
        Returns:
            Number of vectors added
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        if self.normalize:
            vectors = self._normalize_vectors(vectors.astype(np.float32))
        else:
            vectors = vectors.astype(np.float32)
        
        # Train index if needed (for IVF-PQ)
        if self.index_type == "IVF-PQ" and self.index.is_trained == False:
            print("Training IVF-PQ index...")
            train_size = min(100000, len(vectors))
            self.index.train(vectors[:train_size])
        
        # Add vectors
        start_id = self.total_vectors
        self.index.add(vectors)
        
        # Update ID mapping and metadata
        for i, (ext_id, meta) in enumerate(zip(ids, metadata or [{}] * len(ids))):
            internal_id = start_id + i
            self.id_map[internal_id] = ext_id
            self.metadata[ext_id] = meta
        
        self.total_vectors += len(vectors)
        return len(vectors)
    
    def search(self, query: np.ndarray, k: int = 10, 
               filter_fn: Optional[callable] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search for nearest neighbors
        
        Args:
            query: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return
            filter_fn: Optional filter function (id, score, metadata) -> bool
            
        Returns:
            List of (external_id, score, metadata) tuples
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        if self.normalize:
            query = self._normalize_vectors(query.astype(np.float32))
        else:
            query = query.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query, k * 2)  # Search more for filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            ext_id = self.id_map.get(idx)
            if ext_id is None:
                continue
            
            meta = self.metadata.get(ext_id, {})
            
            # Apply filter if provided
            if filter_fn and not filter_fn(ext_id, dist, meta):
                continue
            
            results.append((ext_id, float(dist), meta))
            
            if len(results) >= k:
                break
        
        return results
    
    def batch_search(self, queries: np.ndarray, k: int = 10) -> List[List[Tuple[str, float, Dict]]]:
        """Batch search for multiple queries"""
        if self.normalize:
            queries = self._normalize_vectors(queries.astype(np.float32))
        else:
            queries = queries.astype(np.float32)
        
        distances, indices = self.index.search(queries, k)
        
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            query_results = []
            for dist, idx in zip(query_distances, query_indices):
                if idx == -1:
                    continue
                ext_id = self.id_map.get(idx)
                if ext_id:
                    meta = self.metadata.get(ext_id, {})
                    query_results.append((ext_id, float(dist), meta))
            all_results.append(query_results)
        
        return all_results
    
    def delete_vectors(self, ids: List[str]) -> int:
        """Delete vectors by external IDs (note: FAISS doesn't support deletion, we mark as deleted)"""
        # FAISS doesn't support deletion, so we maintain a blacklist
        if not hasattr(self, '_deleted_ids'):
            self._deleted_ids = set()
        
        count = 0
        for ext_id in ids:
            self._deleted_ids.add(ext_id)
            count += 1
        
        return count
    
    def save(self, path: str):
        """Save index and metadata to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save FAISS index
        index_path = f"{path}.index"
        if self.gpu_id >= 0:
            cpu_index = self.faiss.index_gpu_to_cpu(self.index)
            self.faiss.write_index(cpu_index, index_path)
        else:
            self.faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = f"{path}.meta.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'id_map': self.id_map,
                'metadata': self.metadata,
                'total_vectors': self.total_vectors,
                'deleted_ids': getattr(self, '_deleted_ids', set())
            }, f)
        
        print(f"Index saved to {path}")
    
    def load(self, path: str):
        """Load index and metadata from disk"""
        index_path = f"{path}.index"
        self.index = self.faiss.read_index(index_path)
        
        metadata_path = f"{path}.meta.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.id_map = data['id_map']
            self.metadata = data['metadata']
            self.total_vectors = data['total_vectors']
            if 'deleted_ids' in data:
                self._deleted_ids = data['deleted_ids']
        
        if self.gpu_id >= 0:
            self._move_to_gpu(self.gpu_id)
        
        print(f"Index loaded from {path}")


class BillionScaleIndex:
    """
    Distributed index for billion-scale vector storage
    Uses sharding and hierarchical search
    """
    
    def __init__(self, dimension: int = 768, num_shards: int = 8,
                 index_type: str = "IVF-PQ", gpu_ids: List[int] = None):
        """
        dimension: Embedding dimension
        num_shards: Number of shards for distribution
        index_type: Type of index for each shard
        gpu_ids: List of GPU IDs to use (None for CPU)
        """
        self.dimension = dimension
        self.num_shards = num_shards
        self.index_type = index_type
        self.gpu_ids = gpu_ids or [-1] * num_shards
        
        self.shards = []
        self.shard_assigner = None  # Will be trained on first add
        
        self._initialize_shards()
    
    def _initialize_shards(self):
        """Initialize shard indexes"""
        for i in range(self.num_shards):
            gpu_id = self.gpu_ids[i] if i < len(self.gpu_ids) else -1
            shard = FAISSVectorStore(
                dimension=self.dimension,
                index_type=self.index_type,
                gpu_id=gpu_id
            )
            self.shards.append(shard)
    
    def _assign_shard(self, vectors: np.ndarray) -> np.ndarray:
        """Assign vectors to shards using simple hashing"""
        # Use first dimension for simple sharding (can be improved with learned routing)
        shard_ids = (vectors[:, 0] * 1000).astype(int) % self.num_shards
        return shard_ids
    
    def add_vectors(self, vectors: np.ndarray, ids: List[str],
                    metadata: Optional[List[Dict]] = None) -> int:
        """Add vectors to appropriate shards"""
        shard_ids = self._assign_shard(vectors)
        
        total_added = 0
        for shard_idx in range(self.num_shards):
            mask = shard_ids == shard_idx
            if mask.sum() == 0:
                continue
            
            shard_vectors = vectors[mask]
            shard_ids_list = [ids[i] for i in range(len(ids)) if mask[i]]
            shard_metadata = [metadata[i] for i in range(len(metadata or [])) if mask[i]] if metadata else None
            
            added = self.shards[shard_idx].add_vectors(shard_vectors, shard_ids_list, shard_metadata)
            total_added += added
        
        return total_added
    
    def search(self, query: np.ndarray, k: int = 100) -> List[Tuple[str, float, Dict]]:
        """Search across all shards and return top-k results"""
        all_results = []
        
        # Search in each shard
        for shard in self.shards:
            if shard.total_vectors == 0:
                continue
            results = shard.search(query, k=k // self.num_shards + 10)
            all_results.extend(results)
        
        # Sort by score and return top-k
        all_results.sort(key=lambda x: x[1], reverse=(self.shards[0].metric == "cosine"))
        return all_results[:k]
    
    def save(self, base_path: str):
        """Save all shards"""
        for i, shard in enumerate(self.shards):
            shard.save(f"{base_path}_shard{i}")
    
    def load(self, base_path: str):
        """Load all shards"""
        for i, shard in enumerate(self.shards):
            shard_path = f"{base_path}_shard{i}"
            if os.path.exists(f"{shard_path}.index"):
                shard.load(shard_path)
