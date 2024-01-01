
class Model(torch.nn.Module):
    def __init__(self, conv=torch.nn.ConvTranspose2d):
        super().__init__()
        self.conv_transpose = conv(6, 45, 7, stride=2, padding=1)
    def forward(self,x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
m = Model(torch.nn.ConvTranspose2d)
y1 = m(x1)
