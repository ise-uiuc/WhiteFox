
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other_linear
        return v2

# Initializing the model
m = Model()

# Other tensors that have same size as the linear transformation output
other = torch.randn(1, 4)

# Inputs to the model
x1 = torch.randn(1, 3)
