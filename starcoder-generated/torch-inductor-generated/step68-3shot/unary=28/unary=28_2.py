
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5, bias=False)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(-0.1, 0.1)

# Inputs to the model
x2 = torch.randn(1, 3, 256, 16)
