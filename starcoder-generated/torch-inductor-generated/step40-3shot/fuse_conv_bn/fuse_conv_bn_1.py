
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=4, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        torch.manual_seed(0)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU(inplace=False)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        t1 = self.pool1(self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))))
        t2 = self.relu3(self.bn3(self.conv3(t1)))
        return self.pool3(t2)
# Inputs to the model
torch.manual_seed(0)
x1 = torch.randn(4, 1, 224, 224)
