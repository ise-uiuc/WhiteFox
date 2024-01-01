
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(3, 32, 3, padding=1, stride=2)
        self.conv1 = torch.nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2)
        self.conv2 = torch.nn.ConvTranspose3d(64, 9, 3, padding=1, stride=2)
    def forward(self, x1):
        v0 = x1
        v1 = self.conv(v0)
        v2 = F.leaky_relu(v1, 0.2)
        v3 = self.conv1(v2)
        v4 = F.tanh(v3)
        v5 = self.conv2(v4)
        v6 = F.softmax(v5, dim=-1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32, 32)
