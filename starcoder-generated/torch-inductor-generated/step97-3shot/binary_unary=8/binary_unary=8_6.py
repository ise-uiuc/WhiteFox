
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_bn_relu_1 = torch.nn.Sequential(torch.nn.Conv2d(64, 32, 4, stride=2, bias=False, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.conv_bn_relu_2 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 5, stride=1, bias=False, padding=2), torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.conv_bn_relu_3 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, stride=1, bias=False, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.conv_bn_relu_4 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 5, stride=1, bias=False, padding=2), torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.conv_bn_relu_5 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, stride=1, bias=False, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU())
        self.conv_bn_relu_6 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 5, stride=1, bias=False, padding=2), torch.nn.BatchNorm2d(32), torch.nn.ReLU())
    def forward(self, x6):
        x1 = self.conv_bn_relu_1(x6)
        x2 = self.conv_bn_relu_2(x1)
        x3 = self.conv_bn_relu_3(x2)
        x4 = self.conv_bn_relu_4(x3)
        x5 = self.conv_bn_relu_5(x4)
        x7 = self.conv_bn_relu_6(x5)
        x9 = (self.conv_bn_relu_2(x1) + self.conv_bn_relu_4(x3) + self.conv_bn_relu_5(x4) + self.conv_bn_relu_6(x5))
        x10 = torch.relu(x9) + self.conv_bn_relu_2(x1) + self.conv_bn_relu_4(x3) + self.conv_bn_relu_5(x4) + self.conv_bn_relu_6(x5)
        return x10
# Inputs to the model
x6 = torch.randn(1, 64, 224, 224)
