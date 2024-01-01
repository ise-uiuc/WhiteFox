
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=0.7071067811865476):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min = min_value
        self.max = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 3)
