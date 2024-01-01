
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transposed_conv = torch.nn.ConvTranspose2d(73, 29, kernel_size=(2, 3), stride=(2, 2), padding=(2, 0), bias=False)
    def forward(self, x2):
        o1 = self.transposed_conv(x2)
        o2 = o1 > 0
        o3 = o1 * 0.396
        o4 = torch.where(o2, o1, o3)
        return o4
# Inputs to the model
x2 = torch.randn(3, 73, 17, 9)
