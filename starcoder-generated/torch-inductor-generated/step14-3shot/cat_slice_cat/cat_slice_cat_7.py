
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
 
    def forward(self, x1):
        out1 = self.conv1(x1)
        out2 = self.conv2(out1)
        c1 = torch.nn.functional.interpolate(out2, scale_factor=2, mode='trilinear', align_corners=None)
        c2 = torch.cat([c1, out2], dim=1)
        c3 = c2[:, 100:200]
        c4 = torch.cat([c1, c3], dim=1)
        return c4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
