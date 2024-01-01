
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(1, 8, (3, 3), stride=2, bias=False)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(8, 8, (3, 3), stride=1, bias=False)
        torch.manual_seed(1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        torch.manual_seed(1)
        self.conv3 = torch.nn.Conv2d(8, 8, (3, 3), stride=1, bias=False)
        torch.manual_seed(1)
        self.bn3 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 16, 16)
