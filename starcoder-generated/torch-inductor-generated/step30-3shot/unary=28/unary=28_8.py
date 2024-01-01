
class Model(torch.nn.Module):
    def __init__(self, min_value=0., max_value=0.):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Init. with random config
m = Model(min_value=torch.randn(1, 1), max_value=torch.randn(1, 1))

# Init. with a fixed config
m = Model(min_value=0.485, max_value=0.45)

# Inputs to the model
x1 = torch.randn(1, 8)
