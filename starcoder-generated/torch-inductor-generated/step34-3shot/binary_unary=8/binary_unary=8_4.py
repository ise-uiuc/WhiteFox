
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (3, 5), stride=(2, 3), padding=(1, 2))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avgpool(v1)
        v3 = torch.nn.functional.interpolate(v2, scale_factor=2, mode='nearest', align_corners=False)
        # replace the above line with the one below to fix the issue
        # v3 = torch.nn.functional.interpolate(v2, scale_factor=2, mode='nearest')
        v4 = v1 + v3
        return torch.nn.functional.relu(v4)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
