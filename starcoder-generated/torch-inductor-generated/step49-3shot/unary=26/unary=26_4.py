
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 3, 2, stride=1, padding=0, bias=True)
        self.conv_t2 = torch.nn.ConvTranspose2d(3, 12, 3, stride=1, padding=0, bias=False)
    def forward(self, x4):
        o1 = self.conv_t1(x4)
        o2 = self.conv_t2(o1)
        return o2
# Inputs to the model
x4 = torch.randn(10, 3, 14, 45)
