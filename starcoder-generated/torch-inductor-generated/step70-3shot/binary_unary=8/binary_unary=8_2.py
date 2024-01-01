
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1, bias=False)
        self.conv4 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1, bias=False)
        self.conv5 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv5(x1)
        v1 = v1.permute(0, 1, 3, 2)
        v2 = v2.permute(0, 1, 3, 2)
        v3 = v3.permute(0, 1, 3, 2)
        v4 = v4.permute(0, 1, 3, 2)
        v5 = v5.permute(0, 1, 3, 2)
        v6 = v1 + v2 + v3 + v4 + v5
        v6 = v6.permute(0, 1, 3, 2)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
