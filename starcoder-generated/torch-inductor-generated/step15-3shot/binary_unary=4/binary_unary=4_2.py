
class Model(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.linear = torch.nn.Linear(k, 1)

    def forward(self, x1, other=None):
        if other is None:
            x2 = self.linear(x1)
        else:
            x2 = self.linear(x1) + other
        x3 = torch.nn.functional.relu(x2)
        return x3

# Initializing the model
m = Model(3)

# Inputs to the model
x1 = torch.randn(1, 3)
