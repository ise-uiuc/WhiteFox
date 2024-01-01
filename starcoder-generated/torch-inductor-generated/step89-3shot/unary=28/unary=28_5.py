
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(256, 1000)
 
    def forward(self, x1, min_value=-1, max_value=1):
        v1 = self.layer(x1)
        v3 = torch.clamp_min(v1, min_value)
        v4 = torch.clamp_max(v3, max_value)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
