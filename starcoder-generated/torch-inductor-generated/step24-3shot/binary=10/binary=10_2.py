
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing input tensors
x1 = torch.randn(1, 5)
x2 = torch.randn_like(x1)

# Initializing the model
m = Model()

# Inputs to the model
