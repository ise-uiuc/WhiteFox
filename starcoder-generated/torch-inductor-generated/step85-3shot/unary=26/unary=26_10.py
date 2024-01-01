
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 11, 5, stride=2, padding=0, bias=False)
    def forward(self, x5):
        x= torch.nn.Conv2d(10, 11, 5, stride=2, padding=0, bias=False)
        x12 = self.conv_t(x5)
        x13 = x12 > 0
        x14 = x12 * 5.77
        x15 = torch.where(x13, x12, x14)
        return x15
# Inputs to the model
x5 = torch.randn(11, 10, 25, 22)
