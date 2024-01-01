
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 1, 3, stride=(2, 3), padding=(3, 1), output_padding=(2, 1), groups=3, bias=True)
    def forward(self, x7):
        t1 = self.conv_t(x7)
        t2 = t1 > 0
        t3 = t1 * -0.5707
        t4 = torch.where(t2, t1, t3)
        return torch.nn.functional.adaptive_avg_pool2d(t4, (1, 1))
# Inputs to the model
x7 = torch.randn(20, 3, 54, 51)
