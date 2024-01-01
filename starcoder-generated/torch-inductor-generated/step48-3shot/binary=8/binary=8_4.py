
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 7, stride=2, padding=3)
        self.conv2 = torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 1, 16, stride=8, padding=4, output_padding=0)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.tanh1 = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.tanh2(v5)
        v7 = self.bn1(v6)
        v8 = self.relu4(v7)
        v9 = self.bn2(v8)
        v10 = torch.sigmoid(v9)
        v11 = self.tanh1(v10)
        return v11
# Inputs to the model
x = torch.randn(1, 1, 256, 256)
