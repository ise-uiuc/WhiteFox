
class Model3(torch.nn.Module):
    def __init__(self, min_value=0, max_value=1):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        r = self.linear(x1)
        r = torch.clamp(r, self.min_value, self.max_value)
        return r

# Initializing the model
m1 = Model3()
m2 = Model3(min_value=-4, max_value=-2)

# Inputs to the model
x1 = torch.randn(1, 16)
