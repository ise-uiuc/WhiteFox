
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 1, 8, stride=1, bias=False)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0
        x3 = x1 * -0.26
        x4 = torch.where(x2, x1, x3)
        x5 = x4.neg()
        return torch.nn.functional.relu6(x5)
# Inputs to the model
x3 = torch.randn(1, 9, 14, 15)
