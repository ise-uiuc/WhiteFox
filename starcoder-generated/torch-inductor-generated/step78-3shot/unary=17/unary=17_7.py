
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 4, 7)
        self.conv2 = torch.nn.ConvTranspose2d(4, 8, 5)
        self.conv3 = torch.nn.ConvTranspose2d(8, 4, 3)
        self.conv4 = torch.nn.ConvTranspose2d(4, 16, 7, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = torch.relu(v2)
        v6 = torch.relu(v3)
        v7 = torch.relu(v4)
        v8 = torch.sigmoid(v5)
        v9 = torch.sigmoid(v6)
        v10 = torch.sigmoid(v7)
        return v10
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
