
class Model(nn.Module):
    def __init__(self, m1, m2, m3, m4):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
    def forward(self, x1, x2, x3, x4):
        v1 = self.m1(x1, self.m2)
        v2 = self.m3(x3, self.m4)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)

class Model(nn.Module):
  def __init__(self, A):
    super().__init__()
    self.A = A
  def forward(self, x):
    return self.A.mm(x) # Matrix multiplication between self.A and x

m1 = Model(self.m1)
m2 = self.m2
m3 = Model(self.m3)
m4 = self.m4
