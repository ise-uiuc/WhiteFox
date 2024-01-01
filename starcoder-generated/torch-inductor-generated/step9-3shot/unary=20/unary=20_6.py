
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(3, 16, 2, stride=(2, 2))
        self.conv2 = torch.nn.ConvTranspose2d(16, 32, 2, stride=(1, 2), groups=1, padding=(0, 1), dilation=(1, 1), bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = x1
        v2 = torch.tanh(v1)
        x = v2
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 272, 384)
