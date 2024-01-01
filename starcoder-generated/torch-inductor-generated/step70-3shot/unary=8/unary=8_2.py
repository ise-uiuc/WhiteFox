
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 12, 3, 1, 1)
        )
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
