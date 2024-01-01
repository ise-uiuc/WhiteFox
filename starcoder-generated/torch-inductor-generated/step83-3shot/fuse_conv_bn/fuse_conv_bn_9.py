
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.max_pool = torch.nn.MaxPool2d(stride=2)
        self.conv1 = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.avg_pool = torch.nn.AvgPool2d(2)
    def forward(self, x):
        x = self.bn1(x)
        x = self.max_pool(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.avg_pool(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 5, 5)
