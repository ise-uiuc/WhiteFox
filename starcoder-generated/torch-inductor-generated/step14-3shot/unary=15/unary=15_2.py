
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.bn2(self.conv3(self.bn1(self.conv2(self.conv1(x1)))))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 300, 300)
