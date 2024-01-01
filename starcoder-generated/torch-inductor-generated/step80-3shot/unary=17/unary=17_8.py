
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = self.conv(v3)
        v5 = torch.relu(v4)
        v6 = torch.relu(v5)
        v7 = self.conv(x1[:, -2:, :, :])
        v8 = torch.relu(v7)
        v9 = self.conv(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
