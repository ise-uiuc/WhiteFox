
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 128, 3, stride=1, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, stride=1, padding=1), torch.nn.BatchNorm2d(256), torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, stride=1, padding=1), torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer6 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 3, stride=1, padding=1), torch.nn.ReLU())
        self.layer7 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, stride=1, padding=1), torch.nn.ReLU())
        self.layer8 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer9 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, stride=1, padding=1), torch.nn.ReLU())
        self.layer10 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, stride=1, padding=1), torch.nn.ReLU())
        self.layer11 = torch.nn.Sequential(torch.nn.Conv2d(512, 512, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.layer12 = torch.nn.Sequential(torch.nn.Conv2d(512, 1024, 3, stride=1, padding=1), torch.nn.ReLU())
        self.layer13 = torch.nn.Sequential(torch.nn.Conv2d(1024, 1024, 3, stride=1, padding=1), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = self.layer2(v1)
        v3 = self.layer3(v2)
        v4 = self.layer4(v3)
        v5 = v3 + v4
        v6 = self.layer5(v5)
        v7 = self.layer6(v6)
        v8 = self.layer7(v7)
        v9 = v6 + v8
        v10 = self.layer8(v9)
        v11 = self.layer9(v10)
        v12 = self.layer10(v11)
        v13 = v10 + v12
        v14 = self.layer11(v13)
        v15 = self.layer12(v14)
        v16 = self.layer13(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
