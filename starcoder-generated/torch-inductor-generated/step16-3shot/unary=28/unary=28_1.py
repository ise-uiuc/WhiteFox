
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_max(v1, 1e-12)
        v3 = torch.clamp_min(v2, 1 / 128)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
min_value = 1e-38
max_value = 32
