
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
 
    def forward(self, x1, m1=0.2, M1=0.8):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, m1)
        v3 = torch.clamp_max(v2, M1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
