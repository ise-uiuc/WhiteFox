
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def g(self, x):
        return x
    def forward(self, x):
        y = self.g(x)
        y = torch.cat([y, y], dim=1).view(x.size(0), -1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 3)
