
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (3, 3), stride=1, bias=False, padding=(1, 1), dilation=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 128, (2, 2), stride=2, bias=False, padding=(1, 1), dilation=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        y = self.conv2(x)
        y = self.bn2(y)
        return y
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
