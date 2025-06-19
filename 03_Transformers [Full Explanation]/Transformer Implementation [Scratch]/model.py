import torch
import torch.nn as nn
import math

# Encoder Architecture
"""
Encoder has the architecture consisting of multiple layers that are:
1. Input Embeddings
2. Positional Encodings
3. Multi-Head Self-Attention
4. Add & Layer Normalization
5. Feed Forward Neural Network
6. Add & Layer Normalization
7. Output of the Encoder
The output of the encoder is a set of continuous representations of the input tokens.

In the paper, they have use 6 layers of the encoder, this is the meaning of the Nx in the paper.
The Nx means that the encoder has 6 layers, each layer has the same architecture.
"""

# Layer 1: Input Embeddings
class InputEmbeddings(nn.Module):

    def __init__(self, dimension_model:int, vocabulary_size: int):
        """
        dimension_model: The dimension of the model, typically 512 or 768.
        vocabulary_size: The size of the vocabulary, which is the number of unique tokens in the input data.

        The Embeddings means that it will map the input tokens to a continuous vector space.
        Pytorch provide an `nn.Embedding` layer that does this mapping.
        The `nn.Embedding` layer takes two arguments: the size of the vocabulary and the dimension of the model.
        """
        super().__init__()
        self.dimension_model = dimension_model
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, dimension_model)

    def forward(self, x):
        """
        In the paper, Section 3.4: In the embedding layers, we multiply those weights by sqrt(dimension_model)
        to scale the input embeddings.
        x: The input tokens, which are integers representing the tokens in the input data.
        """
        return self.embedding(x) * math.sqrt(self.dimension_model)
    
# Layer 2: Positional Encodings
class PositionalEncoding(nn.Module):
    
    def __init__(self, dimension_model:int, sequence_length:int, dropout: float) -> None:
        super().__init__()
        self.dimension_model = dimension_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (sequence_length, dimension_model)
        positionalencoding = torch.zeros(sequence_length, dimension_model)
        # Create a vector of shape (sequence_length, 1)
        position = torch.arange(0, sequence_length, dtype = torch.float).unsqueeze(1) # (sequence_length, 1)
        div_term = torch.exp(torch.arange(0, dimension_model,2)).float() * (-math.log(10000.0) / dimension_model)
        # sin is used for even indices and cos for odd indices. We will use twice.
        # Apply sin to even indices
        positionalencoding[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        positionalencoding[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension [so that it can be used in the forward method]
        positionalencoding = positionalencoding.unsqueeze(0) # Shape: (1, sequence_length, dimension_model)
        # Register the positional encoding as a buffer so that it is not considered a parameter
        self.register_buffer('positionalencoding', positionalencoding)

    def forward(self, x):
        """
        Adding the positional encoding to the input embeddings.
        """
        x = x + (self.positionalencoding[:, :x.shape[1], :])
        return self.dropout(x)
    
# Layer Normalization
