
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 1024, 1, stride=1, padding=0)
        self.groupnet1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 1024, 1, stride=1, padding=0),
        )
        self.groupnet2 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(1024, 512, 1, stride=1, padding=0),
        )
        self.groupnet3 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1, 1, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )
        self.groupnet4 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 1, 1, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.groupnet1(v1)
        v3 = self.groupnet2(v2)
        v4 = self.groupnet3(v3)
        v5 = self.groupnet4(v3)
        return v4 * v5
# Inputs to the model
x1 = torch.randn(1, 128, 64, 64)
