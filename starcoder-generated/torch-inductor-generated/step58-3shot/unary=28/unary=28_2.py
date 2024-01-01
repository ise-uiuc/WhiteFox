
class Model(torch.nn.Module):
    def __init__(self, min_value=0.3, max_value=0.5):
        super().__init__()
        self.linear = torch.nn.Linear(3,3)
        self.max_value = max_value
        self.min_value = min_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.empty(1, 3).uniform_(-10., 10.)
     
