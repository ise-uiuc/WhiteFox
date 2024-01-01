
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min = min_value)
        v3 = torch.clamp_max(v2, max = max_value)
        return v3

# Inputs to the model
x1 = torch.randn(1, 64)
min_value = -8
max_value = 8
