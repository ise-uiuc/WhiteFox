
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        v2 = torch.clamp_min(self.linear(x1), min=0.1)
        v3 = torch.clamp_max(v2, max=0.2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
