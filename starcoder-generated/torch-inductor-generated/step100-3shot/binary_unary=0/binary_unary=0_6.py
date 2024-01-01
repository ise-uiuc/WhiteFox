
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        a1 = v1 + v2
        a2 = self.conv3(a1)
        v3 = a1 + a2
        v4 = self.conv2(v3)
        v5 = self.conv3(v4)
        v6 = v4 + v5
        return v6
#Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
#Model ends
