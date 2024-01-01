
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 3072, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, min=-0.987654321)
        v3 = torch.clamp_max(v2, max=0.123456789)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024)
