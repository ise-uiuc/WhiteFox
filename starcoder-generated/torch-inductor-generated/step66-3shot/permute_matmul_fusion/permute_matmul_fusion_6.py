
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(x1.permute(1, 2, 0), x2.permute(2, 0, 1))[0][0][0]
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 1, 2)
