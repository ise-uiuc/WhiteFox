
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin = torch.nn.Linear(256, 8192)

    def forward(self, x1, x2):
        v1 = self.lin(x1)
        v2 = v1 + x2
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 8192)
