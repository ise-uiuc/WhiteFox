
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 16, 5)
        self.conv1 = torch.nn.ConvTranspose2d(16, 4, 5)
        self.conv2 = torch.nn.ConvTranspose2d(4, 1, 2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
