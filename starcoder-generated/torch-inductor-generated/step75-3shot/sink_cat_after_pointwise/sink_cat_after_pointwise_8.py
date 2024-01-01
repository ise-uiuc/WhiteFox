
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def g(self, x):
        return x
    def h(self, x):
        return self.g(x)
    def forward(self, x):
        y = self.h(x)
        y = torch.cat([y, y, y], dim=1).view(y.size(0), -1)
        return torch.relu(y)
# Inputs to the model
x = torch.randn(2, 3)
