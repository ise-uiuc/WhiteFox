
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 6, kernel_size=(1,1), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        self.sigmoid = v1 * v2
        return self.sigmoid
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
