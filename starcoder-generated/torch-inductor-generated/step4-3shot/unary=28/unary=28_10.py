
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.33, max_value=0.33):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model() 

# Inputs to the model
x1 = torch.randn(1, 1)
