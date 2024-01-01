
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.expand_as(torch.zeros_like(x))
        y = y + x
        y = y.tanh()
        y = y.view(3, 2, 2)
        y = y.select(0, 1)
        y = y * y
        return y
# Inputs to the model
x = torch.randn(1, 2, 2)
