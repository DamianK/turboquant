import numpy as np
from scipy.stats import ortho_group

# 1. Setup (Done once)
d = 1024  # Dimension
b = 3     # Bit-width (e.g., 3-bit)
# Precompute a random orthogonal rotation matrix
rotation_matrix = ortho_group.rvs(dim=d) 
# Precompute optimal centroids (simplified for this example)
centroids = np.linspace(-1, 1, 2**b)

def turboquant_compress(x):
    """Compresses a vector using TurboQuant."""
    # Step 1: Rotate the vector
    y = rotation_matrix @ x
    
    # Step 2: Scalar quantization (find nearest centroid indices)
    # In practice, this uses efficient hardware-level lookups
    indices = np.abs(y[:, None] - centroids).argmin(axis=-1)
    return indices

def turboquant_decompress(indices):
    """Decompresses back to original space."""
    # Step 1: Map indices back to centroid values
    y_hat = centroids[indices]
    
    # Step 2: Reverse rotation (using transpose for orthogonal matrix)
    x_hat = rotation_matrix.T @ y_hat
    return x_hat

# Usage
original_vector = np.random.randn(d)
compressed_indices = turboquant_compress(original_vector)
reconstructed_vector = turboquant_decompress(compressed_indices)

