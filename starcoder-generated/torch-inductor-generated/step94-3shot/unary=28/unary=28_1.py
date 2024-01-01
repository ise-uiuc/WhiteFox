 Initialization
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value=min_value)
        v3 = torch.clamp_max(v2, max_value=max_value)
        return v3

# Initializing the model with user-inputted min and max values
m = Model(min_value=2.5,max_value=-3.0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
