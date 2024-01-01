
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()
        self.linear = torch.nn.Linear(8, 64)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x1):
        v1 = self.prelu(self.linear(x1))
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
