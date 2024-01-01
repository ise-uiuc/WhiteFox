
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = x1.permute(0, 2, 1)
        v4 = x2.permute(0, 2, 1)
        v3 = torch.matmul(v2, v4)
        v6 = v3.permute(0, 2, 1)
        v5 = torch.matmul(x2, v6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
