
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = x1 + x1
        v2 = torch.relu(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 64, 64)
x3 = torch.randn(1, 32, 64, 64)

