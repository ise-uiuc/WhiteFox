
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.matmul(torch.matmul(torch.bmm(x1, x1), x2), x2)
# Inputs to the model
x1 = torch.randn(1, 2, 4)
x2 = torch.randn(1, 2, 4)
