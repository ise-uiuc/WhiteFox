
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (5, 5), stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(1024, 2048, (1, 1), stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(512, 256, (4, 4), stride=2, padding=1, bias=False)
        self.conv4 = torch.nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1, bias=False)
        self.conv5 = torch.nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
