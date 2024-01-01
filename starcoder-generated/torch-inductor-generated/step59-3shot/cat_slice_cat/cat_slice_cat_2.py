
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.interpolate(v1,
                            size=(480,640),
                            mode='bilinear',
                            align_corners=False)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:480]
        return torch.cat([v2, v4], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
