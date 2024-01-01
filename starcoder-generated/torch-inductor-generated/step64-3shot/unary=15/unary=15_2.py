
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(192, 1, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        s1 = torch.sigmoid(v1) + torch.sigmoid(v2) + torch.sigmoid(v3)
        s2 = torch.sigmoid(v1) * torch.sigmoid(v2) * torch.sigmoid(v3)
        s3 = torch.sigmoid(v1 + v2 + v3)
        return s1, s2, s3
# Inputs to the model
x1 = torch.randn(5, 64, 28, 28)
x2 = torch.randn(5, 256, 14, 14)
x3 = torch.randn(5, 192, 14, 14)
