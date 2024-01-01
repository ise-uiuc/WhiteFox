
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 512, 3, padding=1, stride=2, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.max_pool2d(v1,2,2)
        v3 = v2.reshape(-1, 512)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 80, 80)
