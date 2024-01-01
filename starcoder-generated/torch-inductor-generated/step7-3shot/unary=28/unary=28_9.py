
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.clamp_min = torch.nn.Hardtanh(0.0, 0.5208095618495932)
        self.clamp_max = torch.nn.Hardtanh(0.5208095618495932, 1.4692311549774012)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = self.clamp_min(v1)
        return self.clamp_max(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
