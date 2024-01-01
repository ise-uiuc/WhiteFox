
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 2, 2, stride=1, groups=2, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 3, 3)
