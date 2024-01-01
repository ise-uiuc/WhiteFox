
class Model(torch.nn.Module):
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.lower = lower_bound
        self.upper = upper_bound
 
    def forward(self, x):
        x1 = self.linear(x)
        x2 = torch.clamp_min(x1, self.lower) 
        x3 = torch.clamp_max(x2, self.upper) 
        return x3

# Initializing the model
lb = -0.8
ub = 0.8 
m = Model(lb, ub)

# Inputs to the model
x3 = torch.randn(1, 3)
