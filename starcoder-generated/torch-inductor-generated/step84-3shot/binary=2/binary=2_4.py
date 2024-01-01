
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
    def forward(selfm, x10):
        v1 = self.conv(x10)
        v2 = v1 - 307.993408203125
        return v2
# Inputs to the model
x10 = torch.randn(1, 3, 64, 64)
