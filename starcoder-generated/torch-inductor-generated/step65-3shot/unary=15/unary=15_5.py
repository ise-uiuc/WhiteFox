
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(64, 128, (5, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.conv2 = torch.nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(128, 256, 3, stride=1, padding=2, dilation=2)
        self.conv4 = torch.nn.Conv3d(256, 256, 3, stride=2, padding=0)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv3d(256, 512, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.conv3(v4)
        v6 = self.relu3(v5)
        v7 = self.conv4(v6)
        v8 = self.relu4(v7)
        v9 = self.conv5(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 6, 16, 16)
