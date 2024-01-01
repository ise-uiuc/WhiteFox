
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = torch.relu(t1)
        t3 = self.conv2(t2)
        t4 = torch.relu(t3)
        t5 = self.conv3(t4)
        t6 = torch.relu(t5)
        t7 = self.conv4(t6)
        t8 = torch.relu(t7)
        t9 = self.conv5(t8)+t8
        t10 = torch.relu(t9)
        t11 = self.conv6(t10)+t10
        t12 = torch.relu(t11)
        v1 = self.conv7(t12)+t12
        return v1
# Inputs to the model
x1 = torch.randn(4, 64, 28, 28)
