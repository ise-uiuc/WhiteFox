
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, groups=2)
        self.conv2_1 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2_1(v2)
        v3 = F.relu(v3)
        v3 = self.conv2_2(v2)
        v3 = F.relu(v3)
        v4 = self.conv3(v3)
        v4 = F.relu(v4)
        v4 = self.conv4(v4)
        v4 = F.relu(v4)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
