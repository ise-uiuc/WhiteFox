
class Model(torch.nn.Module):
  def __init__(self, dim, inv_scale_factor):
    super().__init__()
    self.query = torch.nn.Parameter(torch.randn(dim, dim, dtype=torch.float64, device="cpu") * 0.416, requires_grad=True, name="query")
    self.key = torch.nn.Parameter(torch.randn(dim, dim, dtype=torch.float64, device="cpu") * 0.416, requires_grad=True, name="key")
    self.value = torch.nn.Parameter(torch.randn(dim, dim, dtype=torch.float64, device="cpu") * 0.416, requires_grad=True, name="value")
    self.inv_scale_factor = torch.nn.Parameter(inv_scale_factor, requires_grad=False, name="inv_scale_factor")
    self.dropout_p = 0.4164164164

  def forward(self, dropout_p):
    qk = torch.matmul(self.query, self.key.transpose(-2, -1))
    scaled_qk = qk.div(self.inv_scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(self.value)
    return output

# Initializing the model
m = Model(dim=32, inv_scale_factor=1 / (2 ** 10))

# Inputs to the model
dropout_p = torch.tensor(0.4164164164, dtype=torch.float64, device="cpu")
