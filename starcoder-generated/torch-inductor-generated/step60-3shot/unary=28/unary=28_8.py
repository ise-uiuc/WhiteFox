
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.linear_min = torch.nn.Linear(1, 32)
        self.linear_max = torch.nn.Linear(32, 16)
        self.min_value = min_value
 
    def forward(self, x1):
        v1 = self.linear_min(x1)
        v2 = torch.clamp_min(v1, 0.0)
        v3 = torch.clamp_max(v2, self.min_value)
        return v3

# Initializing the model
m = Model(0.2)

# Initializing the model with an invalid value for the `min_value` argument
## The input tensor of the model is initialized to a random tensor, which doesn't fulfill the constraint
m = Model(-0.1)

# Inputs for the model
## The input tensor is initialized to a random tensor, which satisfies the constraint
x1 = torch.randn(1, 1)
