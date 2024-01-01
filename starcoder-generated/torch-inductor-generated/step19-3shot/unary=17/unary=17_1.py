
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(100, 16, kernel_size=5, stride=2)
        self.conv2 = torch.nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1)
        self.conv3 = torch.nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 100, 32, 32)
