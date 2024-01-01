
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.bn = torch.nn.BatchNorm1d(3, affine=False)
    self.linear = torch.nn.Linear(3, 3, False)
    self.linear2 = torch.nn.Linear(3, 3, False)
  def forward(self, x, inp):
    x1 = self.bn(x)
    v1 = self.linear(x1)
    v2 = self.linear2(v1)
    v3 = v1 + x1
    v4 = torch.mm(v3, v3)
    t1 = v2 + v3
    t1 = v4 + t1
    t2 = t1 + v4
    t2 = torch.mm(t1, t1)
    t2 = t2 + inp
    return (v1,v2,t2, v3.detach() + x)
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
