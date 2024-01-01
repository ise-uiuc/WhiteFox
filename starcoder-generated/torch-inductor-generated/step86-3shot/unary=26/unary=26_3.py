
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(199, 1, 18, stride=6, padding=9, bias=False)
    def forward(self, x0):
        t1 = self.conv_t(x0)
        t2 = t1 > 0
        t3 = t1 * 0.61
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x0 = torch.randn(19, 199, 6, 49)
