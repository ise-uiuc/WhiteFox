
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, other, x1):
        v1 = torch.nn.functional.linear(x1, torch.full([3, 8], 0.25))
        v2 = other + v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.full([1, 8], 0.5)
