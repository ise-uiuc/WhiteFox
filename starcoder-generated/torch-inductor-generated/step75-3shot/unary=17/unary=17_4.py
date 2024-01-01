
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 16, 3, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, 3, stride=2)
        self.conv3 = torch.nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.conv4 = torch.nn.ConvTranspose2d(16, 1, 3, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = F.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
