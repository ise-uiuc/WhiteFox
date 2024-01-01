
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transposed_conv = torch.nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, padding=1, bias=False)
    def forward(self, x1, x2, x3):
        v1 = self.transposed_conv(x1)
        v2 = torch.sigmoid(torch.add(x2, v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 20, 40, 40)
x2 = torch.randn(1, 20, 80, 80)
x3 = torch.randn(1, 20, 160, 160)
