
class Model(torch.nn.Module):
    def __init__(self, min_channels, max_channels):
        super().__init__()
        self.max_channels = max_channels
        self.conv = torch.nn.Conv1d(2, min_channels, 3, stride=1, padding=1)
    def forward(self, x1):
        channels = torch.randint(low=self.max_channels, size=(1,)).item()
        x2 = self.conv(x1)
        y = torch.clamp_max(x2, channels)
        return y
min_channels = 1
max_channels = 1
# Inputs to the model
x1 = torch.randn(1, 2, 12)
