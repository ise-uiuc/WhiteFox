
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv2d(32, 40, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(32, 40, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(40, eps=0.0010000000475, momentum=0.0, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()

    def forward(self, x, x_):
        x1 = self.conv_1(x)
        x_1 = self.conv_2(x_)
        x = torch.add(x1, x_1)
        x = self.relu(x)
        x = self.bn(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 32, 64, 64)
x_ = torch.randn(1, 32, 64, 64)
