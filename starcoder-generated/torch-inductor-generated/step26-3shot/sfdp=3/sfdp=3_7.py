
class Model(torch.nn.Module):
  def __init__(self, num_heads=6):
    super().__init__()
    self.num_heads = num_heads
    self.scale_factor = np.sqrt(self.num_heads)
 
  def forward(self, q, k, v, dropout_p=0.):
    qk = torch.mul(torch.matmul(q, k.transpose(-2, -1)), self.scale_factor)
    softmax_qk = torch.nn.functional.softmax(qk, dim=-1)
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
    output = dropout_qk.matmul(v)
    return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 6, 256, 256)
k = torch.randn(1, 6, 256, 256)
v = torch.randn(1, 6, 256, 256)
dropout_p = 0.1
