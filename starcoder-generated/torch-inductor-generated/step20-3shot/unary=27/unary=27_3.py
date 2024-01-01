
class Model(torch.nn.Module):
    def __init__(self, channel_min_clamp=0.25, channel_max_clamp=0.4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.channel_min_clamp = torch.zeros(8) + channel_min_clamp
        self.channel_max_clamp = torch.zeros(8) + channel_max_clamp
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp(v1, min=self.channel_min_clamp, max=self.channel_max_clamp)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 96, 96)
