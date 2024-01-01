
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, output_padding=1, groups=2)
        self.conv_t2 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x0):
        identity = x0
        x3 = self.conv_t1(x0)
        x4 = self.conv_t2(x3)
        out = x4 - identity
        return out
