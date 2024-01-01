
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtransposeB2D = torch.nn.ConvTranspose2d(64, 64, 7, stride=2)
    def forward(self, x):
        t1 = self.convtransposeB2D(x)
        t2 = torch.sigmoid(t1)
        t3 = t1 * t2
        return t3
# Inputs to the model
x1 = torch.randn(1, 64, 320, 320)
