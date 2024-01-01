
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, 3, stride=1, padding=1)  # NCHW format
        self.conv2 = nn.Conv2d(16, 16, 1, stride=1, padding=0) # NCHW format
    def forward(self, x1):
        v1 = x1.permute(0, 2, 3, 1)
        v2 = self.conv1(v1)
        v3 = v2.permute(0, 3, 1, 2)
        v4 = self.conv2(v3)
        v5 = v4 + 3
        v6 = F.relu(v5)
        v7 = torch.clamp_max(v6, 6)
        v8 = torch.div(v7, 6)
        return v8
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
