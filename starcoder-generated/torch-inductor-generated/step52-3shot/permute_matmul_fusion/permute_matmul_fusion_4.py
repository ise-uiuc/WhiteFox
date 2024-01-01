
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(2, 0, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = x1.permute(1, 0, 2)
        v4 = x1.permute(0, 1, 2)
        v5 = torch.matmul(x2, v1) + torch.matmul(v1, x1) + torch.matmul(x1, v2) + torch.matmul(v2, x1) + torch.matmul(x1, v3) + torch.matmul(v3, x1) + torch.matmul(x1, v4) + torch.matmul(v4, x1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
