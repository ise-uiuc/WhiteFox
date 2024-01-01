
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(512, 128, 8, groups=2, bias=False)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(128, 512, 1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 512, 8, 8)
