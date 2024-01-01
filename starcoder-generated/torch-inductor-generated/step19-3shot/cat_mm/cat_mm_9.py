
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1 ** 2
        v2 = x2 ** 3
        return torch.cat([v1, v2], 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
