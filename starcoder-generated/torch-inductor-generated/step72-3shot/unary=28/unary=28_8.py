
class Model(torch.nn.Module):
    def __init__(self, max_value=127.0, min_value=96.0):
        super().__init__()
        self.linear = torch.nn.Linear(45, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 45)
