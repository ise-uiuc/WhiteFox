
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2, x1, x2, x1, x2], 0)
        v2 = torch.cat([x1, x2, x1, x2, x1, x2], 0)
        return torch.cat([v1, v2], 1)
# Inputs to the model
x1 = torch.randn(2, 7)
x2 = torch.randn(2, 7)
