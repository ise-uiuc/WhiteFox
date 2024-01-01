
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 10)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 100
        v3 = v2 - 200
        v4 = v3 - torch.ones(10) * 300
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
