
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min_v = min_value
        self.max_v = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_v)
        v3 = torch.clamp_max(v2, self.max_v)
        return v3

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
min_val, max_val = 5, 10
