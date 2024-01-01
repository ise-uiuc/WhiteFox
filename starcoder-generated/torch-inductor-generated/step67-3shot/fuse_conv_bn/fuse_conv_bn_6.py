
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        def conv3x3(in_channel, out_channel, stride=1):
            return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1)

        self.conv3x3 = conv3x3(176, 192)
        self.bn = nn.BatchNorm2d(192)
    def forward(self, inputs):
        y = self.conv3x3(inputs)
        o = self.bn(y)
        return o
# Inputs to the model
input = torch.randn(1, 176, 4, 4)
