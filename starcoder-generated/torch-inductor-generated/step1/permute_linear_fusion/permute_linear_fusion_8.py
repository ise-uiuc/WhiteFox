
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x1):
        v1 = x1.permute(1, 0)
        v2 = torch.nn.functional.linear(self.linear(v1), self.linear.weight, self.linear.bias)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
