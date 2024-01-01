
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, 2, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 3, 224, 224)
