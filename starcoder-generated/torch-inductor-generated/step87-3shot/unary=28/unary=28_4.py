
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 20)
 
    def forward(self, x2):
        v1 = self.linear(x2)
        v2 = torch.clamp_min(v1, min=0)
        v3 = torch.clamp_max(v2, max=20)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(100, 1, 1, 1)
