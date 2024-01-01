
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 167, 3, stride=1, padding=1, bias=False)
    def forward(self, x16):
        t1 = self.conv_t(x16)
        t2 = t1 > 0
        t3 = t1 * -0.651877
        t4 = torch.where(t2, t1, t3)
        return torch.nn.functional.pad(t4, (4, 4, 2, 0))
# Inputs to the model
x16 = torch.randn(3, 19, 11, 8)
