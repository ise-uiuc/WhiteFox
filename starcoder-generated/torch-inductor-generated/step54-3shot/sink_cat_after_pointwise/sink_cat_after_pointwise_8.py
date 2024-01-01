
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        x = 2 * y
        y = y.tanh()
        return x - 4 * y
# Inputs to the model
x = torch.randn(3, 5, 2, 2)
