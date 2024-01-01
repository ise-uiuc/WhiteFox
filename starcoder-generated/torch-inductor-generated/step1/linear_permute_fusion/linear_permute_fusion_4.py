
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        v = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        return v.permute(0, 2, 1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 2)
