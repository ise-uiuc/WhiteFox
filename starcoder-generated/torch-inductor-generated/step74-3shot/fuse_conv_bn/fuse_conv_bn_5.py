
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(13, 2, 3, stride=2) # kernel size = 3, pad = 1, stride = 2
        torch.manual_seed(0)
        self.bn = torch.nn.BatchNorm2d(2, affine=True)
        torch.manual_seed(0)
        self.conv2 = torch.nn.Conv2d(2, 34, 2, stride=1) # kernel size = 2, pad = 0, stride = 1
        torch.manual_seed(0)
        self.conv3 = torch.nn.Conv2d(34, 23, 1, stride=1) # kernel size = 1, pad = 0, stride = 1
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 13, 10, 20)
