
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(28, 35, (7, 1), 1, (2, 3))
        self.conv1 = torch.nn.Conv2d(35, 64, (2, 5), 1, (2, 2))
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 35, (1, 7), 1, (0, 2))
        self.conv_transpose1 = torch.nn.ConvTranspose2d(35, 1, (5, 1), 1, (4, 0))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose(v4)
        v6 = torch.relu(v5)
        v7 = self.conv_transpose1(v6)
        v8 = torch.sigmoid(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 28, 7, 100)
