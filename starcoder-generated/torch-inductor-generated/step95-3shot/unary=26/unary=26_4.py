
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv_t = nn.Sequential(
        nn.ConvTranspose2d(480, 192, 3, stride=2, bias=False),
        nn.BatchNorm2d(192), )
    def forward(self, x49):
        y1 = self.conv_t(x49)
        y2 = torch.nn.functional.leaky_relu(y1)
        y3 = torch.nn.functional.adaptive_avg_pool2d(y2, (1, 1))
        return y3
# Inputs to the model
x49 = torch.randn(3, 480, 13, 24)
