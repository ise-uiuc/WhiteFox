
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(16, 25, 9, stride=1, padding=0)
        self.threshold = torch.nn.Threshold(0.6430957031, 0)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.threshold(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 26, 26)
