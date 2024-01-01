
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, dilation=2, bias=False)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
