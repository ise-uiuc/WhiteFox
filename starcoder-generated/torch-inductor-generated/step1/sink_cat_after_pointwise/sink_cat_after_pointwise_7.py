
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        v1 = torch.tanh(x1)
        v2 = torch.tanh(x2)
        z = (torch.cat([v1, v2]) / 2).to(torch.uint8)
        return z

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
