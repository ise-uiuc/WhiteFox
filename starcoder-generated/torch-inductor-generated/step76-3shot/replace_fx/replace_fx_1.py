
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, input):
    return torch.rand_like(input)
# Inputs to the model
input = torch.randn(1, 2, 2)
