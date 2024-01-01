
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.cat([x2] * 64, 1)
        return torch.cat([v, v], 0)
# Inputs to the model
x1 = torch.randn(64, 4)
x2 = torch.randn(4, 4)
