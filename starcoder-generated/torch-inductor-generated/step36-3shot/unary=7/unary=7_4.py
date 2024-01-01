
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(8, 8)
        self.l1 = torch.nn.Linear(8, 8)
    
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 * torch.clamp(v1 + 3, min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
