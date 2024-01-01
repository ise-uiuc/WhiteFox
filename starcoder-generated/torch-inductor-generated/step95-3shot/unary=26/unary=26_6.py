
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(986, 702, 5, stride=2, padding=2, output_padding=1, bias=True)
    def forward(self, x18):
        j1 = self.conv_t(x18)
        j2 = j1 > 0
        j3 = j1 * -8.199585
        j4 = torch.where(j2, j1, j3)
        return torch.nn.functional.adaptive_avg_pool2d(j4, (22, 41))
# Inputs to the model
x18 = torch.randn(7, 986, 13, 20)
