
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 13, 4, stride=2, padding=1, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = torch.where(x2, x1, x1)
        return x3
# Inputs to the model
x = torch.randn(5, 9, 12, 11)
