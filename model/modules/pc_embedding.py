import torch
import torch.nn as nn

class PointCloudEmbedding(nn.Module):
    def __init__(
        self,
        total_embedding_dim: int = 128,# Effective size for the largest point count
        max_point_count: int = 8192,# Maximum number of points in a point cloud
        base_point_count: int = 512,# Number of points in the smallest dimension
        intermediate_dim: int = 64# Intermediate dimension for the embedding
        ):
        super(PointCloudEmbedding, self).__init__()
        self.total_embedding_dim = total_embedding_dim
        self.max_point_count = max_point_count
        self.intermediate_dim = intermediate_dim

        assert max_point_count % base_point_count == 0, "max_point_count must be divisible by base_point_count"
        self.scaling_factor = max_point_count // base_point_count # The scaling between the maximum and minimum point counts
        assert total_embedding_dim % self.scaling_factor == 0, "total_embedding_dim must be divisible by the scaling factor"

        self.base_point_count = base_point_count
        self.base_embedding_dim = total_embedding_dim // self.scaling_factor# The dimension of the base embedding

        self.embedding_layer = nn.Sequential(nn.Linear(3, self.intermediate_dim), 
                                             nn.ReLU(), 
                                             nn.Linear(self.intermediate_dim, self.base_embedding_dim))# 2 linear layers to embed the points
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape# Batch x Points x Channels

        # First embed using the base embedding size
        embedded_x = self.embedding_layer(x)# Batch x Points x base_embedding_dim

        # Now stack points to group all points from the same cluster
        points_to_stack = n // self.base_point_count # There will be base_point_count points after stacking
        embedded_x = embedded_x.view(b, self.base_point_count, self.base_embedding_dim * points_to_stack)# Batch x base_point_count x (base_embedding_dim * points_to_stack)
        # Now repeat the base embedding to match the embedding dimension
        times_to_repeat = self.total_embedding_dim // embedded_x.shape[-1]# The number of times to repeat the base embedding\
        embedded_x = embedded_x.repeat(1, 1, times_to_repeat)
        return embedded_x