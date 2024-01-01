
class Model(torch.nn.Module):
    def __init__(self, linear_weights, linear_bias):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.linear.weight = torch.nn.Parameter(linear_weights)
        self.linear.bias = torch.nn.Parameter(linear_bias)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1
        if (other is not None):
            v2 = v2 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
linear_weights = torch.randn(2, 3)
linear_bias = torch.randn(2)
m1 = Model(linear_weights, linear_bias)

# Input to the model
x1 = torch.randn(1, 3)
