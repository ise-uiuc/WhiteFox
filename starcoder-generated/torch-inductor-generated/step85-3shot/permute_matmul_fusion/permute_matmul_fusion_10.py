
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2, x1):
        v0 = x2.permute(0, 2, 1)
        return torch.matmul(x1, v0)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
