
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)

    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 - 0.3
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 8)
