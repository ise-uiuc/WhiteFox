
class MultiHeadedAttn(nn.Module):
  def __init__(self, model_dim, num_heads):
    super().__init__()
    self.d_k = model_dim // num_heads
    assert model_dim % num_heads == 0
    self._num_heads = num_heads
    self.linears = clones(nn.Linear(model_dim, model_dim), 4)  # clones

  def forward(self, query, key, value, mask=None):
    r