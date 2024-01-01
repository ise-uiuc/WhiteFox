
class Model(torch.nn.Module):
    def __init__(self, input, weight, bias):
        super(Model, self).__init__()
        self.depthconv = torch.nn.Conv2d(input, 3, (3, 3), strides=(1, 1), padding=(1, 1))
        self.pointconv = torch.nn.Conv2d(1, 1, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv0 = torch.nn.Conv2d(3, 3, (3, 3), stride=(2, 2), padding=(1, 1))
        self.norm = torch.nn.BatchNorm2d(3, affine=True)
    def forward(self, x):
        x10 = self.depthconv(x)
        x11 = self.pointconv(x10)
        x9 = self.conv0(x11)
        x12 = self.norm(x9)
        x13 = x10.add(x12)
        return x13
# Inputs to the model
x = torch.randn(20, 3, 32, 32)
