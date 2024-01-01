
class Module1(torch.nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, (3,3), stride=(1,1), padding=(1,1), dilation=(1,1), groups=1, bias=False)
        self.batchnorm2d = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        y1 = self.conv2d(x1)
        y2 = self.batchnorm2d(y1)
        return y2
# Inputs to the model
x1 = torch.randn(1, 6, 4, 4)
