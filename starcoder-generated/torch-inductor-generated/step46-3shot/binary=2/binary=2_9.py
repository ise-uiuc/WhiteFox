
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(330, 688, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 - torch.randn(330, 84, 84, 84)
        return v2
# Inputs to the model
x2 = torch.randn(1, 330, 64, 64)
