
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
        self.linear_bias = torch.nn.Parameter(torch.ones(16))
 
    def forward(self, x1, min_value, max_value):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3 + self.linear_bias

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 8)
min_value = 1.0
max_value = 4.0
