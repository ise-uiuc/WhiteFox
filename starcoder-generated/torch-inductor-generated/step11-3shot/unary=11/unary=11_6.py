
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 1, stride=2, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(64, 16, 3, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, dilation=2)
        self.conv4 = torch.nn.ConvTranspose2d(16, 64, 3, stride=1, padding=1)
        
    def forward(self, x1):
        v1 = self.conv2(self.conv1(x1))
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.conv4(v5)
        v7 = v6 + 3
        v8 = torch.clamp_min(v7, 0)
        v9 = torch.clamp_max(v8, 6)
        v10 = v9 / 6
        v11 = self.conv3(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
