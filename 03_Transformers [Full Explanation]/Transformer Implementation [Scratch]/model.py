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
class LayerNormalization(nn.Module):
    """
    Formula of the Layer Normalization:
    y = alpha * (x - mean) / sqrt(variance + eps) + bias

    Epsilon is a small value to avoid division by zero. The value is typically set to 1e-6.
    The alpha and bias are learnable parameters that are multiplied and added to the output.
    """
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        """
        Notes:
        dim = -1 referes to the last dimension of the tensor.
        """
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        normalized = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return normalized
    
# Feed Forward Neural Network
"""
In the paper "Attention is all you need" mentioned that it is a Fully Connected Layer with ReLU activation function.

FFN(x) = max(0, xW1 + b1)W2 + b2
"""
class FeedForwardBlock(nn.Module):

    def __init__(self, dimension_model: int, dimension_ffn: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dimension_model, dimension_ffn) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dimension_ffn, dimension_model) # W2 and B2

    def forward(self, x):
        # Batch, Sequence, Dimension
        # (batch, sequence_length, dimension_model) -> (batch, sequence_length, dimension_ffn) -> (batch, sequence_length, dimension_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# Multi Head Attention
class MultiHeadAttention(nn.Module):

    def __init__(self, dimension_model: int, head: int, dropout: float) -> None:
        super().__init__()
        self.dimension_model = dimension_model
        self.head = head
        assert dimension_model % head == 0, "dimension_model is not divisible by head"
        self.d_k = dimension_model // head  # Dimension of each head
        self.w_q = nn.Linear(dimension_model, dimension_model) # Weight for query
        self.w_k = nn.Linear(dimension_model, dimension_model) # Weight for key
        self.w_v = nn.Linear(dimension_model, dimension_model) # Weight for value
        # d_v is the dimension of the value, which is the same as d_k
        # In paper, they have used the same dimension for query, key and value.
        self.w_o = nn.Linear(dimension_model, dimension_model) # Weight for output
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (batch, sequence_length, dimension_model) -> (batch, sequence_length, dimension_model)
        key = self.w_k(k)    # (batch, sequence_length, dimension_model) -> (batch, sequence_length, dimension_model)
        value = self.w_v(v)  # (batch, sequence_length, dimension_model) -> (batch, sequence_length, dimension_model)
        # (batch, sequence_length, dimension_model) -> (batch, sequence_length, head, d_k) -> (batch, head, sequence_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.head, self.d_k).transpose(1, 2)  # (batch, head, sequence_length, d_k)
        key = key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1, 2)          # (batch, head, sequence_length, d_k)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1, 2)  # (batch, head, sequence_length, d_k)
