
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
      t0 = x.view(x.shape[0], 1, 1)
      t1 = torch.cat((t0, t0, t0), dim = 1)
      t1 = t1.view(t1.shape[0], -1)
      return torch.relu(t1)
# Inputs to the model
x = torch.randn(1, 1, 1)
