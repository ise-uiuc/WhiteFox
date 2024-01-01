
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 + x1 + x1 + x1 + x1
        v2 = v1 + v1 + v1 + v1 + v1 + v1 + v1 + v1 + v1
        v3 = x1 + x1 + x1 + x1 + x1 + x1
        return torch.cat([v3, v2], 1)
# Inputs to the model
x1 = torch.rand(8, 8)
