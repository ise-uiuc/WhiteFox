
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x = x1.tanh()
        y = x1.tanh()
        x = x.matmul(y)
        return torch.abs(x)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
