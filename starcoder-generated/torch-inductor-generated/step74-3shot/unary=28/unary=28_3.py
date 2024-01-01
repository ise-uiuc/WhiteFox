
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.0, max_value=1.0):
        super().__init__()
        self.linear = torch.nn.Linear(13, 17)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 13)

# Initializing new model with different min_value and max_value
min_value = -0.5
max_value = 0.5
new_m = Model(min_value=min_value, max_value=max_value)
