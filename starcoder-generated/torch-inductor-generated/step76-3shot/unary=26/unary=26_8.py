
class Model():
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn._ConvTransposeNd(1, 2, kernel_size=5, stride=2, padding=2, dilation=3, output_padding=2, groups=2, bias=True, padding_mode='zeros')
        self.relu1 = torch.nn.ReLU()
    def forward(self, x7):
        v0 = self.conv_t(x7)
        v1 = self.relu1(v0)
        return v1
# Inputs to the model
x7 = torch.randn(16, 1, 20, 48, 24)
