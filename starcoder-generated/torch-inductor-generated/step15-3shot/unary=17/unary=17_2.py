
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 2, stride = 1, padding = 0)
    def forward(self, x):
        x = F.relu6(self.conv1(x))
        x = F.relu6(self.conv2(x))
        x = F.relu6(self.conv3(x))
        return x

# Inputs to the model
x = torch.randn(1, 3, 224, 224)
