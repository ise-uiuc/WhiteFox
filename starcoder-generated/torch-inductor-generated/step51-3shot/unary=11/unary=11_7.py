
class Conv2dTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 1, 7, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 1, 7, stride=2)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(x1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(1, 3, 62, 62)
