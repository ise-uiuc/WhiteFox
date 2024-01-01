
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Conv2d(3, out_channels=8, kernel_size=3, padding=1, groups=8)
        self.block0 = torch.nn.Sequential(
            self.m1,
            torch.nn.BatchNorm2d(8, track_running_stats=False)
        )
        self.m2 = torch.nn.Conv2d(8, out_channels=12, kernel_size=1, groups=12)
        self.m3 = torch.nn.Conv2d(12, out_channels=16, kernel_size=3, padding=1, groups=16)
        self.m4 = torch.nn.Conv2d(16, out_channels=20, kernel_size=3, padding=1, groups=20)
        self.m5 = torch.nn.Conv2d(20, out_channels=24, kernel_size=3, padding=1, groups=24)
        self.m6 = torch.nn.Conv2d(24, out_channels=30, kernel_size=5, padding=2, groups=30)
        self.m7 = torch.nn.Conv2d(30, out_channels=32, kernel_size=1, padding=0, groups=32)
        self.block1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(32, track_running_stats=False),
            self.m2,
            self.m3,
            self.m4,
            self.m5,
            self.m6,
            self.m7,
            torch.nn.BatchNorm2d(32, track_running_stats=False)
        )
    def forward(self, x1):
        v1 = self.block0(x1)
        v2 = self.block1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
