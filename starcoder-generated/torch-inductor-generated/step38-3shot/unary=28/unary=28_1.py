
class Model(torch.nn.Module):
    def __init__(self, min_value=-3.0, max_value=10.0):
        super().__init__()
        self.linear = torch.nn.Linear(3, 10)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        self.linear.bias -= torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value=-0.5, max_value=10.0)

# Inputs to the model
x1 = torch.randn(1, 3)
