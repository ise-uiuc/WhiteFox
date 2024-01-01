
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        x11 = torch.nn.functional.interpolate(x1, size=128, scale_factor=1, mode='nearest', align_corners=False)
        v1 = self.conv(x11) #
        x21 = torch.nn.functional.interpolate(x1, size=64, scale_factor=1, mode='nearest', align_corners=False)
        v2 = self.conv(x21) #
        x31 = torch.nn.functional.interpolate(x1, size=32, scale_factor=1, mode='nearest', align_corners=False)
        v3 = self.conv(x31) #
        x41 = torch.nn.functional.interpolate(x11, size=128, scale_factor=1, mode='nearest', align_corners=False)
        v4 = self.conv(x41) #
        v5 = self.conv(x31) #
        v6 = self.conv(x21) #
        v7 = v1 + v2 + v3 + v4 + v5 + v6 #
        v8 = torch.relu(v7) #
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64, requires_grad=True)
