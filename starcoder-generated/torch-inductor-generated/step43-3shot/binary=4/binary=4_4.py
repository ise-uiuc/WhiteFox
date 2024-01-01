
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
        self.linear.weight.data.uniform_(-1.0, 1.0)
        self.linear.bias.data.uniform_(-1.0, 1.0)
        self.other = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
x = torch.randn(1, 3)
m = Model(other=x)

# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
