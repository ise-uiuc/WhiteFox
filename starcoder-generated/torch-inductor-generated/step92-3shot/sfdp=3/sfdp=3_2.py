
class ScaledDotProductAttention(torch.nn.Module):
  def __init__(self, d_k):
    super().__init__()
    self.d_k = d_k

  def forward(self, query, key, value, scale_factor=1., dropout_p=0.):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk * scale_factor
    softmax_qk = F.softmax(scaled_qk, dim=-1)
    dropout = F.dropout(softmax_qk, p=dropout_p)
    return torch.matmul(dropout, value)

# Initializing the model
m = ScaledDotProductAttention(d_k=128)

# Inputs to the model
query = torch.randn(1, 5, 128)
key = torch.randn(1, 10, 128)
value = torch.randn(1, 10, 128)
___
scale_factor = torch.randn(128, 128)
