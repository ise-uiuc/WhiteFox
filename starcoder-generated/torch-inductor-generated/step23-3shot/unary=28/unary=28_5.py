
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(17, 4)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=x2)
        v3 = torch.clamp_max(v2, max_value=0.70710678118654757)
        v4 = v3.sum()
        return v4

# Initializing the model
m = Model(min_value=0.5, max_value=0.7071067811865476)

# Inputs to the model
x1 = torch.randn(1, 17)
x2 = torch.randn(1)
