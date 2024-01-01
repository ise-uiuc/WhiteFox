
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 6)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.clamp_min(v1, 0.4)
        v3 = torch.clamp_max(v2, 0.6)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 12)
