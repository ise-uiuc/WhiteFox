
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.ones([2, 3], dtype=torch.float32), torch.zeros([2, 3], dtype=torch.float32))
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(min_value = 0.05, max_value = 0.8)

# Inputs to the model
x1 = torch.randn(3, 2)
