
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.AvgPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2), nn.AvgPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.AvgPool2d(kernel_size=2))
        self.layer4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2), nn.AvgPool2d(kernel_size=2))
        self.layer5 = nn.Sequential(nn.Linear(24, 100), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.layer7 = nn.Sequential(nn.Linear(100, 100), nn.ReLU())
        self.layer8 = nn.Linear(100, 100)
    def forward(self, x):
        v1 = self.layer1(x)
        v2 = self.layer2(v1)
        v3 = self.layer3(v2)
        v4 = self.layer4(v3)
        v5 = v3.view([-1, 24])
        v6 = self.layer5(v5)
        v7 = self.layer6(v6)
        v8 = self.layer7(v7)
        v9 = self.layer8(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64)
