
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 7, kernel_size=4, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 5, 128, 128)
