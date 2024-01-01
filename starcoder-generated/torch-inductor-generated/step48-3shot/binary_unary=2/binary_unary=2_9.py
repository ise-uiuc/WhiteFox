
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=2, padding=1, dilation=1),
            torch.nn.Conv2d(64, 128, 1, stride=1, padding=0, dilation=1),
            torch.nn.Conv2d(128, 64, 1, stride=1, padding=0, dilation=1)
        )
        self.conv2d = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.layers(x)
        v2 = self.conv2d(x)
        return v1 + v2
x = torch.randn(2, 3, 32, 32)
