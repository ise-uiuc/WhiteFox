
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.upscale = nn.Upsample(scale_factor=2, align_corners=False)
        self.norm1 = nn.InstanceNorm2d(1)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.norm3 = nn.InstanceNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.norm4 = nn.InstanceNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm5 = nn.InstanceNorm2d(64)
        self.conv5 = nn.Conv2d(64, 256, 3, padding=1)
    def forward(self, input_tensor, other1=1,other2=2,other3=3,other4=4,other5=5):
        v1 = self.upscale(input_tensor)
        v2 = self.norm1(v1 + other1)
        v3 = self.conv1(v2)
        v4 = self.norm2(torch.clamp(v3, -1, 1))
        v5 = self.conv2(v4)
        v6 = self.norm3(torch.clamp(v5, -1, 1))
        v7 = self.conv3(v6)
        v8 = self.norm4(torch.clamp(v7, -1, 1))
        v9 = self.conv4(v8)
        v10 = self.norm5(torch.clamp(v9, -1, 1))
        v11 = self.conv5(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
