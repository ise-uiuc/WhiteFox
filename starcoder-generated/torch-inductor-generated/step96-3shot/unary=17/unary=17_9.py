
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 6, 2, bias=False)
        self.conv_2 = torch.nn.ConvTranspose2d(6, 16, 2, bias=False)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
