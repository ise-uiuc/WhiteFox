
class Model(torch.nn.Module):
    def __init__(self, v1):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.min_value = v1
        self.max_value = v2
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
v1 = 0.5
v2 = 0.5
m = Model(v1, v2)

# Inputs to the model
x1 = torch.randn(1, 3)
