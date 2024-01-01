
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = x1.permute(0, 2, 1)
        x4 = x2.permute(0, 2, 1)
        return torch.matmul(x3, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
