
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64, 16)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = v1 * v1 # the squared value
        v2 = v2.clamp(min=0, max=6)
        v3 = v1 + v2
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
