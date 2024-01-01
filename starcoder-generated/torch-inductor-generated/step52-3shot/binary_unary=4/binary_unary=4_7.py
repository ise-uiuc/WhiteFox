
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, other):
        v1 = torch.nn.functional.linear(x1, other)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(128, 256)
