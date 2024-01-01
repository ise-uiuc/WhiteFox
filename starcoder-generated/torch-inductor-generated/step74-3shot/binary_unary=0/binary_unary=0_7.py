
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = x1
        v2 = x2
        v3 = x3
        v4 = x3
        v5 = v3 + x1
        v6 = torch.relu(v5)
        v7 = v2 + v6
        v8 = torch.relu(v7)
        v9 = v4 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
