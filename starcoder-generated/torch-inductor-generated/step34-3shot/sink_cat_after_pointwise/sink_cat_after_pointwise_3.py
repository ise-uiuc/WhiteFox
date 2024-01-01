
class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    t = torch.cat((x, x, x, x), dim=1)
    t.tanh_()
    t = t.view(t.shape[0], -1)
    t.tanh_()
    t = t.view(t.shape[0], -1)
    t.tanh_()
    t = t.view(x.shape[0], -1)
    x = t.tanh_()
    return x
# Inputs to the model
x = torch.randn(3, 128, 29)
