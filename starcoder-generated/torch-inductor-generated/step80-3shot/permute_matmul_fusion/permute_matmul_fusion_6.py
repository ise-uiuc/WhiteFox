
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(x1, x2.permute(0, 2, 1))
# Inputs to the model
x1 = torch.randn(1, 16, 16, 3, 3, 128)
x2 = torch.randn(1, 18, 4, 7, 4, 256)
