
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=min_value)
        v3 = torch.clamp(v2, max=max_value)
        return v3

# Initializing the model with minimum and maximum input values
min_value = 0.5
max_value = 1
m = Model(min_value, max_value)

# Inputs to the model
x1 = torch.randn(1, 3)
