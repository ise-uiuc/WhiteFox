
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(32, 64, kernel_size=(3, 2), stride=1)
        self.conv1 = torch.nn.ConvTranspose2d(32, 3, (4, 8), stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(64, 3, (4, 8), stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(3, 64, kernel_size=(2, 1), stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv1(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.conv3(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 32, 10, 11)
