
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2)
        v2 = torch.matmul(v1, x1)
        v3 = torch.matmul(x2, x1)
        v4 = torch.matmul(v3, v2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
