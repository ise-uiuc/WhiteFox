
class Model(torch.nn.Module):
    def __init__(self, n0: int, n1: int):
        super().__init__()
        self.weight0 = torch.randn(n0, n1)
        self.weight1 = torch.randn(n1, n0)
    def forward(self, x):
        y = self.weight0.matmul(x)
        y = y.softmax(dim=1)
        y = y.matmul(self.weight1)
        y = y.sigmoid()
        return y
# Inputs to the model
x = torch.randn(5, 3)
