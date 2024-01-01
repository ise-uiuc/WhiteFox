
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = torch.nn.Linear(3,6)

    def forward(self, x1):
        x = torch.cat([self.mm(x1), self.mm(x1)], dim=1)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
