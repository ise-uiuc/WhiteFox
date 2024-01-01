
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 128, 16, stride=16, padding=16) # Use valid padding, stride 16
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.conv2 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.relu3 = torch.nn.ReLU(inplace=False)
        self.conv3 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.relu1(self.conv(x1))
        v2 = self.relu2(self.conv1(v1))
        v3 = self.conv2(v2)
        v4 = self.relu3(self.conv3(v3))

        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
