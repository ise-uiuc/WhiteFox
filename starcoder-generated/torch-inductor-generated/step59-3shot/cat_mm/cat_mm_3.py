
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.cat([x, x, x, x, x, x, x, x, x, x, x, x], 0)
        return torch.cat([v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(1, 3)
