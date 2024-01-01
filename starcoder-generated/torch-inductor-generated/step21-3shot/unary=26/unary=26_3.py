
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 2, (3, 4), stride=1, padding=(1, 1), bias=True)
    def forward(self, x1):
        t2 = self.conv_t(x1)
        x3 = t2 > 0
        x4 = t2 + 4.3846
        x5 = torch.where(x3, t2, x4)
        return x5
# Inputs to the model
x1 = torch.randn(6, 1, 6, 8)
