
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(64, 64)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 64)
x2 = torch.randn(64, 64)
