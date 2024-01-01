
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=1, groups=4)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=1, groups=4)
        self.conv3 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=1, groups=4)
        self.conv4 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1, groups=4)
    def forward(self, x1):
        q1 = self.conv1(x1)
        q2 = self.conv2(q1)
        q3 = self.conv3(q1)
        q4 = self.conv4(x1)
        q5 = torch.cat((q2, q3, q4), dim=0)
        return q5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
