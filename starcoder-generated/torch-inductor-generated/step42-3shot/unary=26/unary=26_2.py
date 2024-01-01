
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(33, 26, 1, stride=1, padding=0, bias=True)
    def forward(self, x1):
        s1 = self.conv_t(x1)
        s2 = s1 > 0
        s3 = s1 * -0.500
        s4 = torch.where(s2, s1, s3)
        return torch.nn.functional.adaptive_avg_pool2d(s4, (52, 20))
# Inputs to the model
x1 = torch.randn(41, 33, 5, 19)
