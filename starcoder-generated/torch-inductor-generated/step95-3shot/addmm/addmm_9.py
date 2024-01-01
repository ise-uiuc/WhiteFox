
class Model(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = torch.nn.Linear(n, n)

    def forward(self, x):
        y = torch.matmul(x, x.T)
        z =  torch.matmul(x, x.T)
        return self.linear(y) + z
# Inputs to the model
x = torch.randn(5, 10)
