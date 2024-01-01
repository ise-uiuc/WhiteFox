
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 32, 5, padding=2, stride=2)
        self.conv1 = torch.nn.ConvTranspose2d(32, 32, 1)
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 3, 5, padding=2, stride=2)
        self.conv1_transpose = torch.nn.ConvTranspose2d(3, 32, 1)
        self.conv2 = torch.nn.Conv2d(32, 3, 5, padding=2, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose(v4)
        v6 = torch.relu(v5)
        v7 = self.conv1_transpose(v6)
        v8 = torch.relu(v7)
        v9 = self.conv2(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
