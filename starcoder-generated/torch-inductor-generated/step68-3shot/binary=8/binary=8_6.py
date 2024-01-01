
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1)
    def forward(self, x1, x2, **kwargs): # add kwargs
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        return v1, v4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
x2 = torch.randn(1, 3, 16, 16)
