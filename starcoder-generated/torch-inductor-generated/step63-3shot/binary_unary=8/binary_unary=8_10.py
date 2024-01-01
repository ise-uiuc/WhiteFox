
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 32, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(8, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv1(x1)
        v4 = v1 + v2 + v3
        v5 = self.conv1(x1)
        v6 = self.conv2(x1)
        v7 = self.conv3(x1)
        v8 = v3 + v6 + v5 + v7
        v9 = torch.mean(torch.relu(v8), dim=[2, 3], keepdim=True)
        return v9 + v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
