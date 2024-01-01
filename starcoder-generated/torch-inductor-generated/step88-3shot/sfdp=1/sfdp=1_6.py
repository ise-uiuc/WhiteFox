
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.query = torch.nn.Linear(32, 32)
    self.key = torch.nn.Linear(512, 256)
    self.value = torch.nn.Linear(32, 32)
    
    # Additional parameters
    self.dropout_p = 0.2
    self.scale_factor = math.sqrt(32)

  def forward(self, x1):
    q = self.query(x1)
    k = self.key(x1)
    v = self.value(x1)
    qkt = torch.matmul(q, k.transpose(1, 2))
    scaled_qkt = qkt / self.scale_factor
    softmax_qkt = F.softmax(scaled_qkt, dim=-1)
    dropout_qkt = F.dropout(softmax_qkt, p=self.dropout_p)
    output = torch.matmul(dropout_qkt, v)
    return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 512, 32)
