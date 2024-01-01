
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, (1, 4), stride=1, padding=(1, 1), bias=True)
    def forward(self, x):
        y = self.conv_t(x)
        y1 = y > 0
        return y1
# Inputs to the model
x = torch.randn(1, 3, 11, 13)
