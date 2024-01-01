
class ModelSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 7, stride=1, padding=1, dilation=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=1, groups=32, bias=False)
    def forward(self, x):
        a1 = self.conv(x)
        a2 = torch.sigmoid(a1)
        a3 = self.conv5(a2)
        a4 = torch.sigmoid(a3)
        return a4
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
