
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv_transpose_block = torch.nn.Sequential(torch.nn.ConvTranspose2d(2, 4, 3, stride=3, dilation=2, padding=2), self.relu, torch.nn.ConvTranspose2d(5, 2, 7, stride=4, padding=5))
    def forward(self, x1):
        v1 = self.conv_transpose_block(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 50, 50)
