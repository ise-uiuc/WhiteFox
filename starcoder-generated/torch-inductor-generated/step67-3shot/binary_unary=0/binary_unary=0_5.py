
class Model(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels * 2, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, group=channels)
        self.conv3 = torch.nn.Conv2d(channels * 2, channels * 4, 1)
        self.conv4 = torch.nn.Conv2d(channels * 4, 1280, 1)
        self.head = torch.nn.Linear(1280, 1000)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.mean([2, 3])
        x = self.head(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 224, 224)
