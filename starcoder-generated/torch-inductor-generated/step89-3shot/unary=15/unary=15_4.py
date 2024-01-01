
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=-1)
        self.conv1 = torch.nn.Conv2d(48, 48, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=-1)
        self.conv2 = torch.nn.Conv2d(48, 48, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.avgpool3 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=-1)
        self.conv3 = torch.nn.Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool4 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=-1)
        self.conv4 = torch.nn.Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool5 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=-1)
        self.conv5 = torch.nn.Conv2d(24, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool6 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=-1)
        self.conv6 = torch.nn.Conv2d(8, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool7 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=-1)
        self.conv7 = torch.nn.Conv2d(8, 6, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool8 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=-1)
        self.conv8 = torch.nn.Conv2d(6, 6, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.avgpool9 = torch.nn.AvgPool2d(kernel_size=4, stride=1, padding=-1)
        self.conv9 = torch.nn.Conv2d(6, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    def forward(self, x1):
        v1 = self.conv1(self.avgpool1(x1))
        v2 = self.conv2(self.avgpool2(v1))
        v3 = self.conv3(self.avgpool3(v2))
        v4 = self.conv4(self.avgpool4(v3))
        v5 = self.conv5(self.avgpool5(v4))
        v6 = self.conv6(self.avgpool6(v5))
        v7 = self.conv7(self.avgpool7(v6))
        v8 = self.conv8(self.avgpool8(v7))
        v9 = self.conv9(self.avgpool9(v8))
        v10 = torch.tanh(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 36, 100)
