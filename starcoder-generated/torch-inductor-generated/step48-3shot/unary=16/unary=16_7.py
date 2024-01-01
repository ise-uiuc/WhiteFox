

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(37, 17)

    def forward(self, x5):
        v1 = self.linear(x5)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
x5 = torch.randn(2, 37)
m = Model()

# Inputs to the model
