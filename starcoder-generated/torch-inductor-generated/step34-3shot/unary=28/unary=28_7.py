
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 1000)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=-5)
        v3 = torch.clamp_max(v2, max=5)
        v4 = self.linear(v3)
        return v4

# Initializing the model
m = Model()

# Inputs of the model
x1 = torch.randn(1, 224)
