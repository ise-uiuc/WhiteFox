
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (4, 1), stride=(2, 1), padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, (1, 4), stride=(1, 1), padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0, max=6)
        v5 = torch.mul(v1, v4)
        v6 = v5 / 6
        return v6.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 128, 256)
