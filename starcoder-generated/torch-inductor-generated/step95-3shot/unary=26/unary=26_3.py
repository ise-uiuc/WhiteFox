
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(151, 247, 5, stride=1, padding=1, groups=2, bias=False)
    def forward(self, x23):
        i1 = self.conv_t(x23)
        return torch.nn.functional.adaptive_avg_pool2d(i1, (7, 2))
# Inputs to the model
x23 = torch.randn(19, 151, 12, 19)
