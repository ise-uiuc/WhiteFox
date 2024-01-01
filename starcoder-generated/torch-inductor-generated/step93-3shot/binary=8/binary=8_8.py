
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=1)
        self.conv3= torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4= torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.interpolate = torch.nn.functional.interpolate
    def forward(self, x1, x2):
        v1 = self.maxpool1(x1)
        v2 = self.maxpool1(x2)
        v3 = self.conv1(v1)
        v4 = self.conv1(v2)
        v5 = self.conv2(v1)
        v6 = self.conv2(v2)
        v7 = self.maxpool2(v5)
        v8 = self.maxpool2(v6)
        v9 = v5 + v7
        v10 = torch.cat(v8, v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
