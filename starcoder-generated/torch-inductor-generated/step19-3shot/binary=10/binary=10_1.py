
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, __other_1__)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
