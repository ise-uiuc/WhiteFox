
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 10, bias=False)
        other_weight = torch.randn(10, 16, )
        self.linear.weight.data.copy_(other_weight)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
other = torch.randn(10, 16, )
m = Model()

# Inputs to the model
x = torch.randn(16, 16)
