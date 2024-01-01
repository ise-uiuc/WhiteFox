
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = F.relu(F.interpolate(v1 + v2, scale_factor=2, mode='bilinear', align_corners=False))
        return v3
# Inputs to the model
x = torch.randn(2, 3, 32, 32)
