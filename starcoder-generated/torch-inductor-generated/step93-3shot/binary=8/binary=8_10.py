
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 144, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(144, 3, 1, stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 144, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(144, 3, 1, stride=1, padding=0, bias=False)
        self.conv5 = torch.nn.Conv2d(3, 144, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(144, 3, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1.view(-1, 144, 1, 1))
        v1 = self.conv3(x1)
        v3 = self.conv4(v1.view(-1, 144, 1, 1))
        v1 = self.conv5(x1)
        v4 = self.conv6(v1.view(-1, 144, 1, 1))
        v1 = self.conv1(x1)
        v5 = self.conv2(v1.view(-1, 144, 1, 1))
        v1 = self.conv3(x1)
        v6 = self.conv4(v1.view(-1, 144, 1, 1))
        v7 = v5 + v6
        return v2, v7, v4, v3
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
