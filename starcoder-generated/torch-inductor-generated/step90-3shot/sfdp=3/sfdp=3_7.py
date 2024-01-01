
class Model(torch.nn.Module):
  def __init__(self, dropout_p):
      super().__init__()
      self.dropout_p = dropout_p
 
  def forward(self, query, key, scale_factor, value):
    qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_qk = qk.mul(scale_factor)
    softmax_qk = scaled_qk.softmax(dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
    output = dropout_qk.matmul(value)
    return output

# Initializing the model
m = Model(0.5)

# Inputs to the model
query = torch.randn(5, 10, 32)
key = torch.randn(15, 32, 64)
scale_factor = 0.15643446504211426
value = torch.randn(15, 64, 64)
