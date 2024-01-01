
class Model(torch.nn.Module):
    def __init__(self, min_val, max_val):
    
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=False)
        self.min_val = min_val
        self.max_val = max_val
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_val)
        v3 = torch.clamp_max(v2, self.max_val)
        return v3

# Initializing the model
m = Model(-0.75, 0.1)

# Inputs to the model
x1 = torch.randn(3, 4)
