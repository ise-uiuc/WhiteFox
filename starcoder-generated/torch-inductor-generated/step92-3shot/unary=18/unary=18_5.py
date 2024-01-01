
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=11, stride=1, padding=5)
        self.conv2 = torch.nn.Conv2d(64, 36, kernel_size=11, stride=1, padding=5)
        self.conv3 = torch.nn.Conv2d(36, 52, kernel_size=11, stride=1, padding=5)
        self.conv4 = torch.nn.Conv2d(52, 36, kernel_size=11, stride=1, padding=5)
        self.conv5 = torch.nn.Conv2d(36, 28, kernel_size=11, stride=1, padding=5)
        self.conv6 = torch.nn.Conv2d(28, 18, kernel_size=11, stride=1, padding=5)
        self.conv7 = torch.nn.Conv2d(18, 26, kernel_size=11, stride=1, padding=5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 448, 216)
