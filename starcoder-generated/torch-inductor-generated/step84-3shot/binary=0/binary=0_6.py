
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 4, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=1, affine=True)
    def forward(self, x1, x2, bn_training):
        x = self.conv(x1)
        x1 = x2 + x
        x3 = self.bn(x1)
        outs = None + x
        #outs = torch.transpose(outs, 1, 3)  # This is a good line, to trigger the issue
        return outs
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
bn_training = False
