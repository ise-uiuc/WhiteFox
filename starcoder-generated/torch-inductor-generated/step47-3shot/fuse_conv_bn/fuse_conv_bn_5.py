
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = torch.nn.Conv2d(1, 1, 5) # Not working because of padding
        self.conv_pad = torch.nn.Conv2d(1, 1, 5, padding=2)
        self.bn = torch.nn.BatchNorm2d(1)
        self.pool2d = torch.nn.MaxPool2d(2, padding=1)
    def forward(self, x):
        x = self.conv_pad(x)
        x = self.bn(x)
        x = self.pool2d(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 10, 10)
