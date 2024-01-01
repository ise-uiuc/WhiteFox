
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(101, 125, (1, 2, 5), stride=(3, 1, 1), padding=(4, 0, 5), output_padding=(2, 2, 1), groups=35, bias=True)
    def forward(self, x3):
        out = self.conv_t(x3)
        mask = out > 0
        mul = out * -0.0206
        out = torch.where(mask, out, mul)
        return out.view(-1)
# Inputs to the model
x3 = torch.randn(2, 101, 23, 23, 10)
