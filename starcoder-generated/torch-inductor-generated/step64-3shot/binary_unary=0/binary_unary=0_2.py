
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(32, 16, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.ReLU()(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.ReLU()(v3)
        v5 = torch.cat((v2, v4), dim=1)
        v6 = self.conv3(v5)
        v7 = torch.nn.ReLU()(v6)
        v8 = self.conv4(v7)
        v9 = torch.nn.ReLU()(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64, requires_grad=True)
