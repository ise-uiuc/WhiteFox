
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        # Pad input with values 0
        x2 = F.pad(t1, [1, 1, 1, 1])
        t3 = x2 + 3
        x4 = F.pad(t3, [1, 1, 1, 1])
        t5 = x4 + 3
        x6 = F.pad(t5, [1, 1, 1, 1])
        return x6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
