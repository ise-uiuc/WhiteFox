
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 3, stride=2, padding=1)
        self.conv2d = torch.nn.Conv2d(4, 4, 3, stride=2, padding=1)
        self.relu = torch.nn.ReLU(True)
    def forward(self, x1):
        v1 = self.conv2d(self.conv_transpose(x1))
        v2 = self.relu(v1)
        v3 = v2 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
