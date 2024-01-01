
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(8, 32, 1, stride=1, padding=0)
        self.conv1 = torch.nn.ConvTranspose2d(32, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(64, 128, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(256, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.max(v2, v3)
        v5 = torch.cat((v3, x1), dim=1)
        v6 = torch.max(v5, self.conv2(v4))
        v7 = self.conv3(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
