
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return x4
# Inputs to the model
x1 = (torch.ones((2, 32, 32, 32)), torch.ones((2, 3, 32, 32)), torch.ones((2, 32, 32)))
