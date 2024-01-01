
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        m1 = x1 + x2
        m2 = x2 + x3
        m3 = x1 + x3
        m4 = m1 + m2 + m3
        v1 = self.conv1(m4)
        v2 = self.conv2(m4)
        v3 = self.conv3(m4)
        v4 = v1 + v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
x3 = torch.randn(1, 3, 16, 16)
