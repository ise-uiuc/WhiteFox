
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.ConvTranspose1d(32, 32, 3, padding=1)
        self.conv2d = torch.nn.ConvTranspose2d(32, 32, 3, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1d(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2d(v2)
        v4 = torch.relu(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 64)
x2 = torch.randn(1, 32, 256, 256)
