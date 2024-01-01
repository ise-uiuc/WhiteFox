
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0, max_value=128.0):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        z1 = torch.clamp_min(y1, min_value=min_value)
        z2 = torch.clamp_max(z1, max_value=max_value)
        return z2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 10)
