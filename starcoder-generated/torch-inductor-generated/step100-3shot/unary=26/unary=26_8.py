
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(17, 36, 21, stride=1, padding=0, bias=False)
    def forward(self, x0):
        v1 = self.conv_t(x0)
        v2 = v1
        v3 = v1 > -4.54
        v4 = torch.where(v3, v2, torch.tensor(True))
        return v4
# Inputs to the model
x0 = torch.randn(3, 17, 14, 12)
