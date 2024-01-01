
class Model(torch.nn.Module):
    def __init__(self):
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        return v1.permute(0, 2, 1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2, 2)
