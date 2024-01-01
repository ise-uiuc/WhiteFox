
class Net1(torch.nn.Module)
    def __init__(self, min_value, max_value):        super(Net1, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.conv2d_2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv2d_3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv2d_4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv2d_5 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv2d_8 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, input1):
        x1 = self.relu(self.conv2d_1(x1))
        x2 = self.relu(self.conv2d_2(x1))
        x3 = self.relu(self.conv2d_3(x1))
        x4 = self.relu(self.conv2d_4(x1))
        x5 = self.relu(self.conv2d_5(x1))
        x6 = self.relu(self.conv2d_8(x1))
        y = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        y = self.relu(y)
        y = self.relu(y)
        y = self.relu(y)
        v1 = self.relu(self.conv2d_1(x1))
        v2 = torch.clamp_min(v1, -0.66)
        v3 = torch.clamp_max(v2, max_value=0.3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 1920, 1080)
min_value = 0.0
max_value = -0.4
