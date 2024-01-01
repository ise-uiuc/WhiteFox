
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2_transpose = torch.nn.ConvTranspose2d(4, 12, 3, stride=2, padding=1, dilation=2, groups=2, bias=False)
    def forward(self, x1):
        v1 = self.conv2_transpose(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(64, 4, 32, 32)
