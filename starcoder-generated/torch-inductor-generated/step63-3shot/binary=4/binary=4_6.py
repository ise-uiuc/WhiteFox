
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Linear(64 * 64 * 3, 1024)
 
    def forward(self, x1):
        v1 = torch.flatten(x1, 1)
        v2 = self.conv(v1)
        v3 = v2 + torch.ones_like(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
