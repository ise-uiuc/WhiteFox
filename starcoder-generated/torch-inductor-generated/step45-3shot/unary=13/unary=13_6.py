
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(5, 3)

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.weight, self.bias)
        v2 = torch.sigmoid(v1)
        v3 = v2 * v1
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,5)
