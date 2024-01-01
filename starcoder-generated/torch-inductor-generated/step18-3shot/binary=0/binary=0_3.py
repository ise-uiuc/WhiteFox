
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 256, 3, stride=3, padding=3)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = [0, 0, 0, 0]
        v2 = F.conv2d(v1, weight=other, stride=1, padding=padding1)
        v3 = v2 + other
        return v3
# Inputs to the model
x1 = torch.randn(2, 2, 512, 512)
