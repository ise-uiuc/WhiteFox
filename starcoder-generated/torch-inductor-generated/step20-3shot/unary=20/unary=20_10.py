
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 128, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x = torch.randn(2, 2, 32, 32)
