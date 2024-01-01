
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x3):
        v1 = self.linear(x3)
        v2 = torch.clamp_min(v1, 0.1)
        v3 = torch.clamp_max(v2, 0.9)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 3, 64, 64)
