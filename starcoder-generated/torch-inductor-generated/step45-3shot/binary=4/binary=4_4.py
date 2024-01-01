

class Model(torch.nn.Module):
    def __init__(self, i):
        super().__init__()
        self.linear = torch.nn.Linear(i, 1)

    def forward(self, x2, x3, x4):
        v0 = torch.add(x3, x4)
        v1 = self.linear(x2)
        v8 = torch.add(v1, v0)
        return v8


# Initializing the model
m = Model(i=100)

# Inputs to the model
x2 = torch.randn(1, 100)
x3 = torch.randn(1, 100)
x4 = torch.randn(1, 100)
