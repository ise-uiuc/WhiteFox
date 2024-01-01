
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x3 = torch.matmul(x1, x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
