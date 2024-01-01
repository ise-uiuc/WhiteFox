
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(40, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.min(v1, torch.full_like(v1, 6.0)), min=0.0, max=6.0)
        v3 = v2 / 6.0
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 40)
