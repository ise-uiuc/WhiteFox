
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(torch.matmul(x1, x2.permute(0, 2, 1)), torch.matmul(x1.permute(0, 2, 1), x2))[0][0][1]
# Inputs to the model
x1 = torch.randn(1, 1, 2)
x2 = torch.randn(1, 2, 1)
