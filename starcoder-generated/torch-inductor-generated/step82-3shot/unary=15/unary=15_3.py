
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
    def forward(self, x):
        v1 = F.elu(self.conv1(x))
        v2 = F.elu(self.conv2(v1))
        v3 = F.elu(self.conv3(v2))
        v4 = F.elu(self.conv4(v3))
        v5 = F.elu(self.conv5(v4))
        v6 = self.conv6(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 240, 240)
