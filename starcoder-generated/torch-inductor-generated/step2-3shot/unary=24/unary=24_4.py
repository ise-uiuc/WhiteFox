
class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
  def forward(self, x1):
    x2 = x1 + torch.neg(x1)
    return x2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
