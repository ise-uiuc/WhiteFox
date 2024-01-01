
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.Linear(10, 10)(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return torch.sum(v3)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
