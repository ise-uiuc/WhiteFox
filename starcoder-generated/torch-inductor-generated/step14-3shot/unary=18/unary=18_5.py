
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0 = 1
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, stride=2, padding=1)
        for i in ["v0"]: self.__dict__[i].requires_grad = True
    def forward(self, x1):
        v2 = self.conv1(x1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        v7 = torch.sigmoid(v6)
        v8 = self.conv4(v7)
        v9 = torch.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
