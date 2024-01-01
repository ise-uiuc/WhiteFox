
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.sub(x1, x1)
        v2 = torch.sub(x1, x1)
        v3 = torch.sub(x1, v2)
        v4 = v1 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
