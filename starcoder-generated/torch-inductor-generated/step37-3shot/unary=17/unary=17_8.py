
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, 3, stride=2)
        self.norm1 = torch.nn.BatchNorm2d(32)
        self.conv1 = torch.nn.ConvTranspose2d(32, 32, 3, stride=1)
        self.norm2 = torch.nn.BatchNorm2d(32)
        self.max_pool = torch.nn.MaxPool2d(3, 1, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(32, 64, 3, stride=2)
        self.conv3 = torch.nn.ConvTranspose2d(64, 64, 2, stride=1)
    def forward(self, x1):
        v1 = torch.add(input=self.conv(x1), other=5.5)
        v2 = torch.flatten(self.norm1(v1), 1)
        v3 = torch.relu(self.conv1(v2))
        v4 = torch.transpose(self.norm2(v3), 1, 2)
        v5 = torch.sigmoid(v4)
        v6 = self.max_pool(v5)
        v7 = torch.add(input=self.conv2(v6), other=-0.128)
        v8 = torch.avg_pool2d(v7, 3)
        v9 = torch.relu(self.conv3(v8))
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
