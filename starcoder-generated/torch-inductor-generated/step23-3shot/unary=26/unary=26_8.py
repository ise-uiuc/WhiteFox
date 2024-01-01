
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 2, stride=1)
    def forward(self, input):
        t1 = self.conv_t(input)
        t2 = t1 > -5
        t3 = t1 * (-0.55)
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
input = torch.randn(4, 1, 5, 5)
