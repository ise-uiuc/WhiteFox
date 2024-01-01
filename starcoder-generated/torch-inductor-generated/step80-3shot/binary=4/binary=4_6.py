
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Parameters for the model
linear.weight = torch.nn.init.normal_(linear.weight, mean=0, std=3e-2)
linear.bias = torch.nn.init.constant_(linear.bias, 0.0)

# Inputs to the model
x1 = torch.randn(1, 32)
other = torch.randn(1, 32)
