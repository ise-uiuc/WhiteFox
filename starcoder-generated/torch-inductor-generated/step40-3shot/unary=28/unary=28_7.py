
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()

        self.min_value = min_value
        self.max_value = max_value
 
    def clamp_min(self, x):
        return torch.clamp(x, min=self.min_value)
 
    def clamp_max(self, x):
        return torch.clamp(x, max=self.max_value)
 
    def forward(self, x):
        t1 = self.linear(x)
        return self.clamp_max(self.clamp_min(t1))

# Initializing the model
m = Model(-10, 10)

# Input to the model
x = torch.randn(1, 4, 1, 1)
