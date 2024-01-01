
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(102, 91, 5, stride=1, padding=1, groups=47, bias=False)
    def forward(self, x8):
        out = self.conv_t(x8)
        mask = out > 0
        mul = out * -0.51
        out = torch.where(mask, out, mul)
        out = torch.nn.functional.dropout(out)
        return torch.nn.functional.elu(out)
# Inputs to the model
x8 = torch.randn(16, 102, 7, 9)
