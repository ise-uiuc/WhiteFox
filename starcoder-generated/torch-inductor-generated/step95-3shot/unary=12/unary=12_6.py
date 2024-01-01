
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.cat([v1, v1, v1], dim=1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
