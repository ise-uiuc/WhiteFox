
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.min = min
        self.max = max
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1,self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3

# Initializing the model
m = Model(min, max)

# Inputs to the model
