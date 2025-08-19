import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ExpertModule(nn.Module):
  """
  Expert Network: Simplified MLP Version
  """
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.fnc1 = nn.Linear(input_size, hidden_size)
    self.fnc2 = nn.Linear(hidden_size, output_size)
    self.activation = nn.ReLU()

  def forward(self, x):
    x = self.fnc1(x)
    x = self.activation(x)
    x = self.fnc2(x) # Add the second linear layer
    return x

class MoEGating(nn.Module):
  """
  Gating network that determines which experts to use for each input
  """
  def __init__(self, input_size, num_experts, top_k = 2):
    super().__init__()
    self.input_size = input_size
    self.num_experts = num_experts
    self.top_k = top_k
    # Gating weights for expert selection
    self.weight = nn.Parameter(torch.empty(num_experts, input_size))
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def forward(self, x):
    x_flat = x.view(-1, self.input_size)

    # Compute expert scores using linear projection
    scores = F.linear(x_flat, self.weight)
    scores = torch.sigmoid(scores)

    #top-k experts
    top_k_scores, top_k_indices = torch.topk(scores, k = self.top_k, dim = -1)

    # Normalize
    top_k_scores = top_k_scores / (top_k_scores.sum(dim = -1, keepdim = True) + 1e-10)

    return top_k_indices, top_k_scores

class MixtureofExperts(nn.Module):
  """
  Complete MoE Model
  """
  def __init__(self, input_size, hidden_size, output_size, num_experts, top_k = 2):
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k

    # expert networks
    self.experts = nn.ModuleList([
        ExpertModule(input_size, hidden_size, output_size)
        for _ in range(num_experts)
    ])

    # gating networks
    self.gate = MoEGating(input_size, num_experts, top_k)

  def forward(self, x):
    # Get expert assignments and weights from gating network
    expert_indices, expert_weights = self.gate(x)
    batch_size, seq_len, _ = x.shape
    x_flat = x.view(-1, x.shape[-1])

    final_output = torch.zeros(
        (batch_size * seq_len, self.experts[0].fnc2.out_features),
        device = x.device,
        dtype = x.dtype
    )

    for k in range(self.top_k):
      # Get expert indices and weights for current k
      current_indices = expert_indices[:, k]
      current_weights = expert_weights[:, k]

      for i in range(self.num_experts):
        mask = (current_indices == i)
        if mask.any():
          expert_input = x_flat[mask]
          expert_output = self.experts[i](expert_input)
          # Add weighted output to final result
          final_output[mask] += expert_output * current_weights[mask].unsqueeze(-1)
    output = final_output.view(batch_size, seq_len, -1)
    return output

def test_moe():
  input_size = 64
  hidden_size = 128
  output_size = 32
  num_experts = 4
  batch_size = 16
  seq_len = 8
  model = MixtureofExperts(
      input_size = input_size,
      hidden_size = hidden_size,
      output_size = output_size,
      num_experts = num_experts
  )
  x = torch.randn(batch_size, seq_len, input_size)
  output = model(x)

  print("\nRunning Basic Tests....")
  print(f"Input Shape: {x.shape}")
  print(f"Output Shape: {output.shape}")
  assert output.shape == (batch_size, seq_len, output_size), "Output shape mismatch"

  print("\nTesting Gating Network...")
  expert_indices, expert_weights = model.gate(x)
  print(f"Expert indices shape: {expert_indices.shape}")
  print(f"Expert weights shape: {expert_weights.shape}")
  assert expert_indices.shape == (batch_size * seq_len, model.top_k), "Expert indices shape mismatch"
  assert expert_weights.shape == (batch_size * seq_len, model.top_k), "Expert weigths shape mismatch"

  print("\nTesting weight normalization...")
  weight_sums = expert_weights.sum(dim = - 1)
  assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol = 1e-6), "Weights don't sum to 1"
  print("Weight normalization test passed")

  print("\nAll test cases Passed....")

if __name__ == "__main__":
  test_moe()