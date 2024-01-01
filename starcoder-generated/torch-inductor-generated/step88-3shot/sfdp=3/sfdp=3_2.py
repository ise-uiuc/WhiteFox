
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.query_fc = torch.nn.Linear(3, 8)
    self.key_fc = torch.nn.Linear(3, 8)
    self.value_fc = torch.nn.Linear(3, 8)
    self.softmax = torch.nn.Softmax()
    self.dropout = torch.nn.Dropout(0.1)
    self.scale_factor = torch.tensor(3.5, requires_grad=False).cuda()
 
  def forward(self, x1, x2):
    q = self.query_fc(x1)
    k = self.key_fc(x2)
    v = self.value_fc(x2)
    qk = torch.matmul(q, k.transpose(-2, -1))
    qk_scaled = qk.mul(scale_factor)
    dropout_qk = self.dropout(self.softmax(qk_scaled))
    output = dropout_qk.matmul(v)
    return output

# Initializing the model
x3 = torch.randn(3, 3)
x4 = torch.randn(2, 3)
m = Model()
