
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        m1 = self.conv1(x1)
        m2 = self.conv2(x2)
        m3 = m1 + m2
        m4 = self.conv3(x1)
        m5 = self.conv4(x2)
        m6 = m4 + m5
        m7 = self.conv5(x1)
        m8 = self.conv1(x2)
        m9 = m7 + m8
        m10 = m3 + m9
        return m10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
