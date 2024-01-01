
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = torch.nn.Linear(16, 8)
        self.other = torch.nn.Linear(16, 8)

    def forward(self, x1):
        x1 = self.Linear(x1)
        x2 = self.other(x1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
