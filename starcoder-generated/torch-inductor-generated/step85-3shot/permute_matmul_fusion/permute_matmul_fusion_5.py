
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(0, 2, 1)
        v1 = torch.matmul(v0, x1.permute(0, 2, 1))
        v2 = torch.matmul(v0, x1.permute(2, 1, 0))
        v3 = torch.matmul(v1, x2.permute(0, 2, 1))
        v4 = torch.matmul(v2, x2.permute(0, 2, 1))
        v5 = torch.matmul(v3, x2.permute(0, 2, 1))
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
