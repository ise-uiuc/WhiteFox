
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(64, eps = 0.1, momentum = 0.1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(64, eps = 0.001, momentum = 0.001)
        self.conv5 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.batchnorm3 = torch.nn.BatchNorm2d(64, eps = 0.1, momentum = 0.1)
        self.conv7 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(1, 1, 16, stride=1, padding=0)
        
    def forward(self, x):
        v1 = x # Save input 'x'
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.batchnorm1(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = self.batchnorm2(v6)
        v8 = self.conv5(v7)
        v9 = self.conv6(v8)
        v10 = self.batchnorm3(v9)
        v11 = self.conv7(v10)
        v12 = self.conv8(v11)
        v13 = v12 - 0.5
        return v13
        
# Inputs to the model
x = torch.randn(1,1,32,32)
