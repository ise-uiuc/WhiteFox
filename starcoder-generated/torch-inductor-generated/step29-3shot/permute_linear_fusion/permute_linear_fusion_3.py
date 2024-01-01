
class myModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(2, 1)
  def forward(self, x1):
    v1 = x1
    v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
    v3 = -v1
    v3 = v3 * v3
    v4 = v3.mean(dim=0)
    v4 = v4.mean(dim=-1)
    v4 = v4.norm(p=2)
    return v4
# Inputs to the model
x1 = torch.randn(2, 2, 2)
