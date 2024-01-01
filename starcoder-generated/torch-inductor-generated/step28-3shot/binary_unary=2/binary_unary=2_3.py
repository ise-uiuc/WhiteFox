
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 32, 1, 1, 0)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, 1, 0)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.conv4 = torch.nn.ConvTranspose2d(64, 96, 3, stride=2, padding=1, output_padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(96, 2, 4, stride=2, padding=1, output_padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.relu2(v5)
        v7 = self.conv4(v6)
        v8 = self.conv5(v7)
        v9 = self.sigmoid(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 129, 129)
