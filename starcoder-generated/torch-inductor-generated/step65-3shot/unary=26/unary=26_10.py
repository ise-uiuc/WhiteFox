
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 4, 5, stride=3, padding=0, bias=True)
    def forward(self, x1):
        x0 = self.conv_t(x1)
        x1 = x0 > 0
        x2 = x0 * -0.208927
        x3 = torch.where(x1, x0, x2)
        x4 = torch.neg(x3)
        x5 = torch.nn.functional.relu6(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 5, 7, 30)
