
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
        self.conv_t = torch.nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_t(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
