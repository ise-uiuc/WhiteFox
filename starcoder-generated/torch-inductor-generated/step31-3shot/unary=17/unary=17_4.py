
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 16, 2, stride=2, padding=3)
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, 3, stride=3, padding=4)
        self.conv3 = torch.nn.ConvTranspose2d(8, 4, 4, stride=4, padding=0)
        self.conv4 = torch.nn.ConvTranspose2d(4, 2, 4, stride=4, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = torch.sigmoid(v6)
        v8 = self.conv4(v7)
        v9 = torch.tanh(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
