
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(11, 31, 7, stride=2, output_padding=2, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(31, 21, 5, stride=1, padding=1, output_padding=0, bias=False)
    def forward(self, x):
        x1 = self.conv_t1(x)
        x2 = self.conv_t2(x1)
        x3 = x1 > 0
        x4 = x1 * -0.85
        x5 = torch.where(x3, x1, x4)
        x6 = torch.where(x2, x2, x5)
        return x6
# Inputs to the model
x = torch.randn(1, 11, 25, 46)
