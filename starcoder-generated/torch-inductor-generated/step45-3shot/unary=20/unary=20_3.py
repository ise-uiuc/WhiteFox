
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu_t = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x1):
        v1 = self.prelu_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 48, 48)
