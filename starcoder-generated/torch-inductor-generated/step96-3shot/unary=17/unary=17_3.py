
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(2, 1, 3, padding=1, stride=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.max(v1, 1, keepdim=False)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 128, 128)
