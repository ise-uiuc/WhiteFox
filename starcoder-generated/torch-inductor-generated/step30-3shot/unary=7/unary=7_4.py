
class Model(torch.nn.Module):
    def __init__(self):
        from torch import nn
        super().__init__()
        self.linear = nn.Linear(100, 10)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x1):
        l1 = self.linear(x1)
        mx = self.maxpool(l1)
        l2 = mx.clamp(min=0, max=6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 28, 28)
