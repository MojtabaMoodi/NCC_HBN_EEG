"""
ArcFace Loss Implementation for Large-Scale Classification

ArcFace (Additive Angular Margin Loss) is designed for large-scale face recognition
and is highly suitable for user identification tasks with many classes (3000+).

Key advantages:
1. Angular margin increases inter-class distance
2. Feature and weight normalization improves training stability
3. Better discrimination for large number of classes
4. Proven effective for face recognition (10K+ classes)

Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) for large-scale classification.
    
    ArcFace adds an angular margin to the angle between features and class weights,
    which increases inter-class distance and improves discrimination for many classes.
    
    Args:
        num_classes: Number of classes
        embedding_dim: Dimension of feature embeddings (before final FC layer)
        margin: Angular margin in radians (default: 0.5, equivalent to ~28.6 degrees)
        scale: Feature scale parameter (default: 64.0, standard for ArcFace)
        easy_margin: If True, uses easier margin computation (default: False)
    """
    
    def __init__(self, 
                 num_classes: int,
                 embedding_dim: int,
                 margin: float = 0.5,
                 scale: float = 64.0,
                 easy_margin: bool = False):
        super(ArcFaceLoss, self).__init__()
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # Create weight matrix for class centers (will be initialized by model)
        # Shape: (num_classes, embedding_dim)
        # CRITICAL: Use proper initialization for ArcFace weights
        # Xavier uniform can cause model collapse (all predictions to same class)
        # Best practice: Use He/Kaiming initialization for better diversity
        # Since weights are normalized in forward pass, initialization scale matters less
        # but diversity is crucial to prevent collapse
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        # Use He initialization (Kaiming uniform) for better weight diversity
        # This prevents model collapse where all predictions go to one class
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Validate initialization: ensure weights have sufficient diversity
        # Check that weight norms are not all identical (which would cause collapse)
        weight_norms = torch.norm(self.weight, p=2, dim=1)
        norm_std = weight_norms.std().item()
        if norm_std < 1e-6:
            raise RuntimeError(
                f"ArcFace weight initialization failed: weight norms have insufficient diversity "
                f"(std={norm_std:.2e}). This will cause model collapse. "
                f"Check initialization method."
            )
        
        # Pre-compute trigonometric values of margin for efficiency
        # These are used in the forward pass for numerical stability
        self.cos_m = float(torch.cos(torch.tensor(margin)).item())
        self.sin_m = float(torch.sin(torch.tensor(margin)).item())
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: Feature embeddings (before final FC layer)
                       Shape: (batch_size, embedding_dim)
            labels: Class labels
                   Shape: (batch_size,)
        
        Returns:
            Loss value (scalar tensor)
        """
        # Validate inputs
        if embeddings.dim() != 2:
            raise ValueError(f"embeddings must be 2D (batch_size, embedding_dim), got shape {embeddings.shape}")
        if labels.dim() != 1:
            raise ValueError(f"labels must be 1D (batch_size,), got shape {labels.shape}")
        if embeddings.size(0) != labels.size(0):
            raise ValueError(f"Batch size mismatch: embeddings {embeddings.size(0)} vs labels {labels.size(0)}")
        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: got {embeddings.size(1)}, expected {self.embedding_dim}")
        
        # Validate label range
        if labels.max() >= self.num_classes or labels.min() < 0:
            invalid_labels = labels[(labels >= self.num_classes) | (labels < 0)]
            raise ValueError(
                f"Invalid labels detected: {invalid_labels.cpu().numpy()}. "
                f"Labels must be in range [0, {self.num_classes-1}], but found values outside this range. "
                f"This indicates a data inconsistency issue."
            )
        
        # Normalize embeddings (L2 normalization)
        # This is critical for ArcFace - features must be on unit hypersphere
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Normalize weight matrix (L2 normalization)
        # Each class center (weight vector) is normalized to unit length
        # CRITICAL: Ensure weight has same dtype as embeddings for mixed precision
        # The weight parameter is stored in float32, but embeddings may be float16 from autocast
        weight = F.normalize(self.weight.to(dtype=embeddings.dtype), p=2, dim=1)
        
        # Compute cosine similarity between embeddings and class weights
        # Shape: (batch_size, num_classes)
        cosine = F.linear(embeddings, weight)
        
        # Clamp cosine values to [-1, 1] for numerical stability
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Get target cosine values (for the true class of each sample)
        # Shape: (batch_size, 1)
        target_cosine = cosine.gather(1, labels.view(-1, 1))
        
        # Compute sin(theta) from cos(theta) using identity: sin^2 + cos^2 = 1
        # sin(theta) = sqrt(1 - cos^2(theta))
        # We need to handle numerical stability carefully
        target_sin = torch.sqrt(1.0 - torch.clamp(target_cosine.pow(2), min=0.0, max=1.0))
        
        # Apply angular margin using trigonometric identity:
        # cos(theta + margin) = cos(theta) * cos(margin) - sin(theta) * sin(margin)
        # This is more numerically stable than computing theta and then cos(theta + margin)
        # CRITICAL: Convert cos_m and sin_m to tensors with same dtype as target_cosine
        # to avoid dtype mismatches with mixed precision training
        cos_m_tensor = torch.tensor(self.cos_m, dtype=target_cosine.dtype, device=target_cosine.device)
        sin_m_tensor = torch.tensor(self.sin_m, dtype=target_cosine.dtype, device=target_cosine.device)
        target_cosine_margin = target_cosine * cos_m_tensor - target_sin * sin_m_tensor
        
        # Create output logits
        # For target class: use cos(theta + margin)
        # For other classes: use cos(theta)
        # CRITICAL: Ensure dtype consistency for scatter_ operation
        # With mixed precision, cosine and target_cosine_margin may have different dtypes
        # Convert both to the same dtype before scatter_ to avoid dtype mismatch errors
        output = cosine.clone()
        # Ensure target_cosine_margin has the same dtype as output
        target_cosine_margin = target_cosine_margin.to(dtype=output.dtype)
        output.scatter_(1, labels.view(-1, 1), target_cosine_margin)
        
        # Apply scale factor
        # This amplifies the logits, making the margin more effective
        output = output * self.scale
        
        # Compute cross-entropy loss
        # Note: We don't apply softmax here - CrossEntropyLoss does it internally
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        
        return loss
    
    def compute_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from embeddings (for inference).
        
        Args:
            embeddings: Feature embeddings
            
        Returns:
            Logits for all classes
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # CRITICAL: Ensure weight has same dtype as embeddings for mixed precision
        weight = F.normalize(self.weight.to(dtype=embeddings.dtype), p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        # CRITICAL: Convert scale to tensor with same dtype as cosine
        scale_tensor = torch.tensor(self.scale, dtype=cosine.dtype, device=cosine.device)
        logits = cosine * scale_tensor
        return logits
