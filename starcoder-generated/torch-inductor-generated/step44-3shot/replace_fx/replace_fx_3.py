
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(4, 2)
  def forward(self, x1):
    a1 = self.linear(x1)
    a2 = torch.rand(1, 2, 2)
    return F.dropout(a1, p=0.5) * a2
x1 = torch.randn(1, 2, 4)
