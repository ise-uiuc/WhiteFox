
class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.m = nn.BatchNorm2d(15)
    def forward(self, x8):
        y1 = x8
        y2 = y1.type(torch.FloatTensor)
        y3 = y1 + 10.0
        y4 = y3.type(torch.DoubleTensor)
        return y2 - y4
# Inputs to the model
x8 = torch.randn(1, 15, 32, 16)
