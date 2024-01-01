
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = torch.max_pool2d(v2, 3, stride=2, padding=1)
        v4 = v4 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)
