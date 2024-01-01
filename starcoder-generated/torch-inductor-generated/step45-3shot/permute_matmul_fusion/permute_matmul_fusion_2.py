
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1, 3)
        v2 = x2.permute(0, 3, 1, 2)
        v3 = torch.matmul(torch.matmul(v1, v2), x2)
        return v3.permute(0, 2, 3, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2).requires_grad_(True)
x2 = torch.randn(1, 2, 2, 2).requires_grad_(True)
