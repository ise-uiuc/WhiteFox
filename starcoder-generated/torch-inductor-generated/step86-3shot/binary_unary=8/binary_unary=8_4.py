
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), stride=1, bias=False, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, (3, 3), stride=1, bias=False, padding=1)
        self.depth_conv = torch.nn.Conv2d(16, 3, (1, 1), stride=1, bias=False, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 16, (1, 1), stride=1, bias=False, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, (1, 1), stride=1, bias=False, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.depth_conv(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = v1 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
