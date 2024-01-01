
class Model(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = x[:, :268435455]
        x = x[:, :int(32 * 32 * 512)]
        x = torch.cat([x1, x], 1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = 0
if x3 == 0:
  x3 = x1
