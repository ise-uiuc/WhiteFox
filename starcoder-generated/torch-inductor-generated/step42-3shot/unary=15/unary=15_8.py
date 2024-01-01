
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_11 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv_12 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_11(x1)
        v2 = self.conv_12(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
