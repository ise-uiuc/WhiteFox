
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        def conv3x3(in_channel, out_channel, stride=1):
            return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1)
        self.convs = nn.Sequential(
            conv3x3(176, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        x = self.convs(input)
        return x
# Inputs to the model
x = torch.randn(1, 176, 4, 4)
