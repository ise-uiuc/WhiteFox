
class Model(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2


# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)
other = torch.ones(1, 2, 4, 4)
m = Model(torch.nn.Linear(4, 5))
