
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 12)
 
    def forward(self, x1, minimum, maximum):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, minimum)
        v3 = torch.clamp_max(v2, maximum)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
minimum = 0.5
maximum = 2.
