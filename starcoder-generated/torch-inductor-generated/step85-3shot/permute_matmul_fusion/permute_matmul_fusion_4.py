
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        q = x2.permute(0, 2, 1)
        v3 = torch.matmul(x1, q)
        v2 = torch.matmul(x1, v3)
        v1 = x1.permute(0, 2, 1)
        v0 = torch.matmul(v1, v2)
        return v0
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
