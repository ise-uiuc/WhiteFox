
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=2)
        self.conv5 = torch.nn.Conv2d(8, 8, 3, stride=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.zeros_like(v2)

        t1 = self.conv3(v1)
        t2 = self.conv4(t1)
        t3 = v2 + t2

        t4 = self.conv5(t3)
        t5 = self.conv4(t4)
        v4 = v3 + t5

        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 57, 57)
x2 = torch.randn(1, 3, 57, 57)
