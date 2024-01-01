
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 1, stride=1, padding=0)
        self.activation = torch.nn.ReLU(inplace=False)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_fcn1 = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.activation2 = torch.nn.LeakyReLU(negative_slope=0.1,inplace=False)
        self.avgpool2 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(64, 2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.activation(v1)
        v3 = self.avgpool(v2)
        v4 = self.avgpool_fcn1(v3)
        v5 = self.conv2(v2)
        v6 = self.activation2(v5)
        v7 = self.avgpool2(v6)
        v8 = torch.flatten(v7, 1)
        v9 = self.fc(v8)
        return v9 
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
