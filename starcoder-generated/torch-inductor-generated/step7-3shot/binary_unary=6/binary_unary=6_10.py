
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1, bias=True)

    def forward(self, x1, x2):
        t1 = self.linear(x1)
        t2 = t1 - x2
        return torch.nn.ReLU()(t2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1)
