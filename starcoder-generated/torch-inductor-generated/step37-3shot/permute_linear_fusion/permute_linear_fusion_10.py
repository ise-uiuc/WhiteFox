
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(2, 2, 2, groups=2)
    def forward(self, x):
        x = x.view(x.shape[0] * 2, 1, x.shape[2], x.shape[3])
        x = self.conv(x)
        return x.view(x.shape[0] // 2, x.shape[1] * x.shape[2] * x.shape[3])
# Inputs to the model
x = torch.randn(1, 2, 2)
