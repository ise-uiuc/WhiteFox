
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        input = torch.cat((y, x), dim = 1)
        weights = torch.ones_like(input)
        y = input - weights
        return torch.tanh(y)
# Inputs to the model
x = torch.randn(3, 3, 2)
y = torch.randn(3, 3, 2)
