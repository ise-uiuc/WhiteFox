
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.add(x1, x1)
        v2 = torch.add(x1, x1)
        v3 = torch.add(x1, x1)
        v4 = torch.add(x1, x1)
        v5 = torch.add(x1, x1)
        v6 = torch.add(x1, x1)
        v7 = torch.add(x1, x1)
        v8 = v1 + v2 + v3 + v4 + v5 + v6 + v7
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 256, 14, 14)
