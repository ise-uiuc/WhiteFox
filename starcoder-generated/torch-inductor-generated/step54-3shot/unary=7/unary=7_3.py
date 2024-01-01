
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 * torch.clamp(min=0, max=6, v1 + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
