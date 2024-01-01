
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 3, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        m1 = self.conv1.forward(x1)
        m2 = self.conv2.forward(x2)
        m3 = self.conv3.forward(m2)
        m4 = m1 + m2
        v4 = m4 + m3
        return
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
