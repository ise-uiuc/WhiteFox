
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(640, 19, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.conv_t1 = torch.nn.ConvTranspose2d(19, 19, kernel_size=4, stride=2, padding=4, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.conv_t1(v1)
        v2 = torch.sigmoid(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 640, 320, 320)
