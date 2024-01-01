
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.clamp = torch.nn.Hardtanh()

    def forward(self, x):
        v = self.linear(x)
        v = v + 3.
        v = self.clamp(v)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
