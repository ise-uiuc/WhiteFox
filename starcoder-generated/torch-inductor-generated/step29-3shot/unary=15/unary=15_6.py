
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 256, 3, dilation=1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 32, 3, dilation=1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, dilation=1, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 64, 1, stride=2, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t3 = self.conv3(t2)
        t4 = self.conv4(t3)
        t5 = self.conv5(t4)
        t6 = self.conv6(t5)
        t7 = torch.tanh(t5)
        t8 = torch.relu(t7)
        v1 = torch.tanh(t6)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 256, 128)
