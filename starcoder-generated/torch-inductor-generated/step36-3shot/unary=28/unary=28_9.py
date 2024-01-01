
class Model(torch.nn.Module):
    def __init__(self, min_value_, max_value_):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
        self.min_value = min_value_
        self.max_value = max_value_
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(-0.5, 1)

# Inputs to the model
x1 = torch.randn(1, 5) # The input tensor
