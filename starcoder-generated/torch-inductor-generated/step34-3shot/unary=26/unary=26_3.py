
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 16, 5, stride=1, padding=0, bias=False)
        self.conv1 = torch.nn.Conv2d(16, 1, 1)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = self.conv1(v1)
        return v2
# Inputs to the model
x = torch.randn(6, 1, 50, 100)
