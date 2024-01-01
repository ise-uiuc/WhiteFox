
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(180, 67, 2, stride=1, padding=0, bias=False)
    def forward(self, x2):
        w1 = self.conv_t(x2)
        x2 = w1 > 0
        x3 = w1 * -24.7089
        x4 = torch.where(x2, w1, x3)
        return torch.nn.functional.relu6(x4)
# Inputs to the model
x2 = torch.randn(1, 180, 40, 41)
