
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.t2 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.t3 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = self.t2(v1)
        v3 = self.t3(v2)
        v4 = v3 + 3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 16, 16)
