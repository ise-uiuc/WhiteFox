
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 4, 5, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 4, 7, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(x1)
        t3 = self.conv3(x1)
        t4 = self.conv4(x1)
        t5 = t1 + t2 + t3 + t4
        t6 = t5 / 4
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
