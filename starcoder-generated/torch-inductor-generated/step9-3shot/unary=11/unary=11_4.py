
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(True)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
