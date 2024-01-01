
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtr = torch.nn.ConvTranspose1d(32, 94, 93, stride=48, padding=13, bias=False, output_padding=46)
    def forward(self, x40):
        s1 = self.convtr(x40)
        s2 = s1 > 0
        s3 = s1 * 0.196
        s4 = torch.where(s2, s1, s3)
        return s4
# Inputs to the model
x40 = torch.randn(14, 32, 281)
