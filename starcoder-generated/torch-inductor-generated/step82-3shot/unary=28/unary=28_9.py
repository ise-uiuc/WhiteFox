
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.linear = torch.nn.Linear(3, 16, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=self.min_value)
        v3 = torch.clamp_max(v2, max=self.max_value)
        return v3


# Input to the model
x1 = torch.randn(1, 3, 16, 16)

# Initializing min_value and max_value as constants
min_value = 0.0
max_value = 0.001
 
# Creating model
m = Model(min_value, max_value)

# Initializing the model
