
class Model(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1)

    def forward(self, x):
        return torch.tanh(self.conv(x))
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
