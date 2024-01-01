
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ConvTranspose2d(225, 257, 1, padding=0, stride=1)
        self.conv1 = torch.nn.ConvTranspose2d(257, 225, 1, padding=0, stride=1, dilation=3)
        self.conv2 = torch.nn.ConvTranspose2d(225, 225, 1, padding=0, stride=1)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = self.conv0(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv1(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv2(v5)
        v7 = torch.tanh(v6)
        v8 = torch.max_pool2d(v7, 2, stride=1)
        return v8
# Inputs to the model
x1 = torch.randn(1, 225, 16384, 1)
