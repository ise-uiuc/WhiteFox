
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 12, (1, 4), stride=1, padding=(1, 1), bias=False)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * 5.398
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(4, 3, 10, 20)
