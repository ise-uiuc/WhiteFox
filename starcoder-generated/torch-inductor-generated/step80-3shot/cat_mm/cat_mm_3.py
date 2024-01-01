
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mul(x1, x2)
        v2 = []
        for _ in range(x2.size(0)):
            v2.append(v1)
        return torch.cat([v1, v2], 1)
# Inputs to the model
x1 = torch.randn(9, 9)
x2 = torch.randn(9, 1)
