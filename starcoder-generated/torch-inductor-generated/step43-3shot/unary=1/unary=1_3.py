
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(5, 1),
        )

    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.layers(torch.cat([x1, x2, x3, x4, x5]))
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 5)
x3 = torch.randn(1, 5)
x4 = torch.randn(1, 5)
x5 = torch.randn(1, 5)

