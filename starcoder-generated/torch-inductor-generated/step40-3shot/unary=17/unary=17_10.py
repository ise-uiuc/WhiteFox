
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.ConvTranspose2d(3, 64, 3, stride=2)
        self.conv_2 = nn.ConvTranspose2d(64, 128, 3, stride=2)
        self.conv_3 = nn.ConvTranspose2d(128, 256, 3, stride=2)
        self.conv_4 = nn.ConvTranspose2d(256, 512, 3, stride=2)
        self.conv_5 = nn.ConvTranspose2d(512, 1, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = F.relu(v1)
        v3 = self.conv_2(v2)
        v4 = F.relu(v3)
        v5 = self.conv_3(v4)
        v6 = F.relu(v5)
        v7 = self.conv_4(v6)
        v8 = F.relu(v7)
        v9 = self.conv_5(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
