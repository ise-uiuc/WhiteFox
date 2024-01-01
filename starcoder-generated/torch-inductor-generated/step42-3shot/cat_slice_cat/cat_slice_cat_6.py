
class Model(torch.nn.Module):
    def __init__(self):
      pass

    def forward(self, t1, t2, size):
        t3 = t1[:, 0:size]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
