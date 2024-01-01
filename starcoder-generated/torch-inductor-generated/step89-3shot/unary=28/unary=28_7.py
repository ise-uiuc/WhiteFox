
class Model(torch.nn.Module):
    def __init__(self, min_value = 0, max_value = 10):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3

 # Initializing the model with user defined keyword arguments
m = Model(**{'min_value': 0,'max_value': 10})

# Inputs to the model
x1 = torch.randn(1, 10)
