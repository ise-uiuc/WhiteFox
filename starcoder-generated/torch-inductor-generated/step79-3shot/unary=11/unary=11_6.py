
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Sequential(torch.nn.ConvTranspose2d(10, 32, 1, stride=1, padding=0), torch.nn.ReLU(), torch.nn.ConvTranspose2d(32, 64, 3, stride=1, padding=0), torch.nn.ReLU(), torch.nn.ConvTranspose2d(64, 128, 3, stride=1, padding=0), torch.nn.ReLU(), torch.nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(2, 10, 64, 64)
