
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.line = torch.nn.Linear(1024, 512)

    def forward(self, x1):
        v1 = self.line(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
