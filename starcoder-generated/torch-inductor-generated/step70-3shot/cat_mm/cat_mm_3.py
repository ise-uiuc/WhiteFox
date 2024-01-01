
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.tanh(x1), x1 + x1], 1)
        v2 = torch.cat([torch.relu(x2), torch.relu(x2 + v1)], 1)
        v3 = torch.cat([x2, x2 + v1, torch.sin(x2 + v1), torch.cos(x2 + v1)], 1)
        v4 = torch.cat([x2, x2 + v1, torch.cat([x2 + v2, x2 + v2], 1)], 1)
        v5 = torch.cat([x2, x2 + v1, torch.cat([x2 + v3, x2 + v3, x2 + v3], 1)], 1)
        v6 = torch.cat([x2, x2 + v1, torch.cat([x2 + v4, x2 + v4, x2 + v4, x2 + v4], 1)], 1)
        v7 = torch.cat([x2, x2 + v1, torch.cat([x2 + v5, x2 + v5, x2 + v5, x2 + v5, x2 + v5, x2 + v5], 1)], 1)
        v8 = torch.cat([v6, x2, x2 + v1, torch.cat([x2 + v6, x2 + v6, x2 + v6, x2 + v6, x2 + v6, x2 + v6, x2 + v6], 1)], 1)
        return v8
# Inputs to the model
x1 = torch.randn(32, 8)
x2 = torch.randn(32, 4)
