
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, dilation=2)
        self.conv1 = torch.nn.Conv2d(8, 8, 3, stride=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 4, 3, stride=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(4, 2, 3, stride=1, dilation=1)
        self.conv5 = torch.nn.ConvTranspose2d(2, 1, 3, stride=1, dilation=1, padding=2)
        self.max_pool = torch.nn.MaxPool2d(4, 4, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv5(v4)
        v6 = torch.relu(v5)
        v7 = self.max_pool(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
