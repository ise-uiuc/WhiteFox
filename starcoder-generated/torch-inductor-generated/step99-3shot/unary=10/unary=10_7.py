
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.Tensor(2, 3))
    self.bias = torch.nn.Parameter(torch.Tensor(2))
    torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
 
  def forward(self, x1):
    v1 = torch.addmm(self.bias, x1, self.weight.t())
    v2 = v1 + 3
    v3 = torch.clamp_min(v2, 0)
    v4 = torch.clamp_max(v3, 6)
    v5 = v4 / 6
    return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
