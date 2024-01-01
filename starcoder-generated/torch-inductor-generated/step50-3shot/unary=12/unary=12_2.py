
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 4, stride=2, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + torch.sigmoid(v1)
        v3 = v2 / torch.sin(v2)
        return v3
# Inputs to the model
x1 = torch.ones(1, 3, 8, 8)
