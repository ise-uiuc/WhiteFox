
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        h1 = self.conv(x1)
        h2 = h1 + 3
        h3 = torch.clamp_min(h2, 0)
        h4 = torch.clamp_max(h3, 6)
        h5 = h1 * h4
        h6 = h5 / 6
        o1 = self.relu6(h6)
        return o1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
