
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 2, stride=4, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 1, 1, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 1, 1, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(3, 1, 1, stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = (self.conv2(x) - self.conv3(x)) + self.conv4(x)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(1, 3, 17, 17)
