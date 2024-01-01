
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=2, padding=2, bias=False)
    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * -20.128
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x4 = torch.randn(3, 4, 35, 42, device='cuda')
