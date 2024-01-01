
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 512, 7, stride=1, padding=1)
    def forward(self, x1, padding1=1, padding2=1):
        v1 = self.conv(x1)
        v2 = v1 + padding1
        return v2
# Inputs to the model
x1 = torch.randn(1, 256, 84, 84)
