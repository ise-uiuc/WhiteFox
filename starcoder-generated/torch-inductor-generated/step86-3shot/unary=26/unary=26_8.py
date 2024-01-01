
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 9, 4, stride=2, padding=1, output_padding=1, bias=True)
    def forward(self, x2):
        t1 = self.conv_t(x2)
        t2 = t1 > 0
        t3 = t1 * 0.08
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x2 = torch.randn(26, 10, 8, 8)
