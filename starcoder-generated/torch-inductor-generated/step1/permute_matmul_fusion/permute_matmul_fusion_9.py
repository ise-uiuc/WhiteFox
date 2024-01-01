
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([1], dtype=torch.float))

    def forward(self, x1, x2):
        x3 = torch.bmm(x1, x2.permute(0, 2, 1))
        x3 = torch.matmul(x1, x2.permute(0, 2, 1))
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
