"""
Sentence-BERT embeddings with MPS acceleration
"""

import torch
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for Sentence-BERT embeddings with MPS support"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "auto"
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: Sentence-Transformers model name
            device: Device to use ('auto', 'mps', 'cuda', or 'cpu')
                    'auto' will choose MPS if available, then CUDA, then CPU
        """
        self.model_name = model_name
        
        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("MPS (Metal Performance Shaders) is available - using GPU acceleration")
            elif torch.cuda.is_available():
                self.device = "cuda"
                logger.info("CUDA is available - using GPU acceleration")
            else:
                self.device = "cpu"
                logger.info("GPU not available - using CPU")
        else:
            self.device = device
            
        # Initialize SentenceTransformer model
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info(f"Model loaded on device: {self.device}")
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            normalize_embeddings: Whether to normalize embeddings to unit length
            
        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Encode using SentenceTransformer
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def cosine_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine distance between two embeddings
        
        Args:
            embedding1: First embedding (1D array)
            embedding2: Second embedding (1D array)
            
        Returns:
            Cosine distance (1 - cosine similarity), in range [0, 2]
        """
        # Ensure embeddings are 1D
        embedding1 = np.asarray(embedding1).flatten()
        embedding2 = np.asarray(embedding2).flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 1.0
            
        cosine_similarity = dot_product / (norm1 * norm2)
        
        # Convert to distance (1 - similarity)
        cosine_distance = 1.0 - cosine_similarity
        
        return float(cosine_distance)
    
    def pairwise_distances(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute pairwise cosine distances between sets of embeddings
        
        Args:
            embeddings1: First set of embeddings (shape: [n1, dim])
            embeddings2: Second set of embeddings (shape: [n2, dim])
                        If None, compute distances within embeddings1
            
        Returns:
            Distance matrix (shape: [n1, n2] or [n1, n1] if embeddings2 is None)
        """
        from scipy.spatial.distance import cdist
        
        if embeddings2 is None:
            embeddings2 = embeddings1
            
        # Compute pairwise cosine distances
        distances = cdist(embeddings1, embeddings2, metric='cosine')
        
        return distances
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model='{self.model_name}', device='{self.device}', dim={self.get_embedding_dim()})"

