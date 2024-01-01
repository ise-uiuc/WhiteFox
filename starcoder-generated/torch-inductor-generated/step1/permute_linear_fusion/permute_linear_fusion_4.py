
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1, 2)
