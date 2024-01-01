
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.f = torch.nn.functional.linear
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        v1 = self.f(x, self.weight, self.bias)
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model(weight, bias)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
