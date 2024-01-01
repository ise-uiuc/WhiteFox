
class Model(torch.nn.Module):
    def __init__(self, in_planes, planes, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(in_planes, planes)
        self.min_value = 0.0
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(5, 6, 10.0)

# Inputs to the model
x1 = torch.randn(1, 5)
