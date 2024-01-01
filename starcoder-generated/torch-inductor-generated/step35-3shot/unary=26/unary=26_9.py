
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 32, 6, stride=3, padding=2)
        self.conv = torch.nn.Conv2d(32, 17, 6, stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = self.conv(v1)
        v3 = x < v1
        v4 = torch.where(v3, v2, x)
        return v4
# Inputs to the model
x = torch.randn(4, 1, 20, 20, device='cuda')
