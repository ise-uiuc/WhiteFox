
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv_1 = torch.nn.ConvTranspose2d(4, 4, 3, stride=1, padding=0, output_padding=2)
        self.tconv_2 = torch.nn.ConvTranspose2d(4, 4, 4, stride=1, padding=0, output_padding=13)
    def forward(self, x1):
        out = self.tconv_1(x1)
        out = torch.tanh(out)
        out = self.tconv_2(out)
        out = torch.tanh(out)
        return out
# Inputs to the model
x1 = torch.randn(1, 4, 12, 12)
