
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, min=0.2)
        v3 = torch.clamp_max(v2, max=0.8)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
