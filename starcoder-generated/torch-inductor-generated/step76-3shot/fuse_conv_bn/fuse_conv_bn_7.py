
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 5),
            torch.nn.BatchNorm2d(3)
        )
        self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x = self.layer1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
