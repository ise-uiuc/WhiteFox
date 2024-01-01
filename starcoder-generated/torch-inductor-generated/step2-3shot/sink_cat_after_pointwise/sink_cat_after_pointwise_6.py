
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.split(x1, 2, dim=1)
        v2 = torch.split(x2, 2, dim=1)
        v3 = torch.cat(v1 + v2, dim=1)
        v4 = torch.cat([x2, x2], dim=1)
        v5 = v3 + v4
        v6 = torch.cat([v5, v4], dim=1)
        v7 = torch.cat([v3, v6], dim=1)
        v8 = torch.relu(v7)
        v9 = torch.reshape(v8, -1)
        return v9
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(1, 16)
