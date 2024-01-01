
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v0 = torch.mm(x, x)
        v1 = torch.mm(x, x)
        return torch.cat([v0, v1, v1, v0, v1, v0], 1)
# Inputs to the model
x = torch.randn(1, 2)
