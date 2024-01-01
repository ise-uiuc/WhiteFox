
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = torch.rand((1, 3, 64, 64), dtype=torch.float)

    def forward(self, inp, x3):
        x1 = torch.nn.functional.linear(inp, self.weight1)
        x2 = x1 + x3
        x4 = torch.nn.functional.relu(x2)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
inp = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
