
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 6)
        self.linear2 = torch.nn.Linear(3, 6)

    def forward(self, x1):
        x2 = self.linear1(x1)
        x3 = self.linear2(x1)
        x4 = x2 + x3
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
