
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3)
 
    def forward(self, x1):
        h1 = self.conv(x1)
        v1, h1 = h1.split([32, 32], dim=1)
        _, h2 = h1.split([16, 16], dim=3)
        _, h3 = h2.split([8, 8], dim=2)
        x2, h4 = torch.split(h3, [4, 4], dim=2)
        x3, _ = torch.split(h4, [2, 2], dim=3)
        x4 = torch.cat([x1, x2, x3], dim=1)
       return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
