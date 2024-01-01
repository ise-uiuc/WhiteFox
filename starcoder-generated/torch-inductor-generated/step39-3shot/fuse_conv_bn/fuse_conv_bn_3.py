
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 3),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        x = self.conv(x)
        return self.conv2(x)
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
