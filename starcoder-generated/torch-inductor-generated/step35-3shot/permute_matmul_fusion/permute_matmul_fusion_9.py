
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.matmul(x2, x1).permute(0, 2, 1)
        return torch.matmul(v1, v2)
# Inputs to the model
x1 = torch.randn(2, 5, 8)
x2 = torch.randn(2, 8, 5)
