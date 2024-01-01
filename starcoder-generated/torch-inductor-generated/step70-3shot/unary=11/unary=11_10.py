
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 1, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        f1 = torch.floor(v1 + 3)
        v2 = torch.clamp_min(f1, 0)
        f2 = torch.floor(v2 / 6)
        v3 = torch.clamp_min(f2, 0)
        v4 = F.relu(v3)
        v5 = v4 + 3
        return v5
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
