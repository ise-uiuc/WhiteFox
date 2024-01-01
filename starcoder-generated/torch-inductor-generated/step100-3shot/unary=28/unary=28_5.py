
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, __parameters__={'min_value': None,'max_value': None}):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, __parameters__['min_value'])
        v3 = torch.clamp_max(v2, __parameters__['max_value'])
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
