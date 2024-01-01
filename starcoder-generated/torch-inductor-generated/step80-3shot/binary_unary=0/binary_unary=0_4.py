
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        v1 = torch.add(x1, 10)
        v2 = torch.add(v1, 0)
        v3 = torch.add(v2, 1.23)
        v4 = torch.add(v3, 10)
        v5 = torch.add(v4, 4.8)
        v6 = torch.add(v5, 1000)
        v7 = torch.add(v6, 1000000)
        v8 = torch.add(v7, 12345)
        v9 = torch.add(v8, 0)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
