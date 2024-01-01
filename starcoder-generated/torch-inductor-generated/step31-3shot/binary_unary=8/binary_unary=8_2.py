
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 3, 1)
        v2 = x1.permute(0, 2, 3, 1)
        v3 = x1.permute(0, 2, 3, 1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        v6 = v5.permute(0, 3, 1, 2)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
