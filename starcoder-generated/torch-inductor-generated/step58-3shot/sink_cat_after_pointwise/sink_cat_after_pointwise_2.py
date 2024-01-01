
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
      t1 = torch.cat([x, x, x], dim=1)
      t2 = t1.view(x.shape[0], -1)
      t3 = torch.tanh(x)
      t4 = torch.relu(t2)
      t5 = t3.relu()
      return x
# Inputs to the model
x = torch.randn(2, 3, 4)
