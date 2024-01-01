
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv1(x1)
        v4 = self.conv2(x1)
        v5 = torch.relu(v1)
        v6 = torch.relu(v2)
        v7 = torch.relu(v3)
        v8 = torch.relu(v4)
        f1 = torch.cat([v5, v6, v7, v8], 1)
        f2 = torch.nn.functional.interpolate(f1, scale_factor=0.25, mode='bicubic', align_corners=False)
        p1 = f2[:, :8, :, :]
        p2 = f2[:, 8:16, :, :]
        p3 = f2[:, 16:24, :, :]
        p4 = f2[:, 24:32, :, :]
        return (p1, p2, p3, p4)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
