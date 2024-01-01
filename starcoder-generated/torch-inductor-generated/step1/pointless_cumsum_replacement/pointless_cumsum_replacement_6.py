
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x, x2):
        v1 = self.conv(x)
        v2 = v1 + v2
        v3 = torch.cumsum(v2.to(dtype=torch.int), dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64).to(torch.float)
