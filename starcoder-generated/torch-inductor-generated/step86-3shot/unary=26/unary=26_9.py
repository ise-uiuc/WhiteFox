
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(339, 260, 3, stride=1, padding=1, output_padding=1, bias=False)
    def forward(self, x2):
        f1 = self.conv_t(x2)
        f2 = f1 > 0
        f3 = f1 * 0.84
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.adaptive_avg_pool2d(f4, (16, 5))
# Inputs to the model
x2 = torch.randn(5, 339, 4, 24)
