
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 1, 11, stride=2, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, 11, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * (torch.tensor([0.5], dtype=torch.float32,))
        v3 = v1 * (torch.tensor([0.7071067811865476], dtype=torch.float32,))
        v4 = torch.erf(v3)
        v5 = v4 + (torch.tensor([1], dtype=torch.float32,))
        v6 = v2 * v5
        v7 = self.conv2(v6)
        return v7
# Inputs to the model
x1 = torch.randn(7, 32, 56, 5);
