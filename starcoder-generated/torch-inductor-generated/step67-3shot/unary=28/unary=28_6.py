
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6, bias=False)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return x3

# Initializing the model
m = Model(min_value=-5.727494, max_value=1.342664)

# Inputs to the model
x1 = torch.randn(1, 6)
