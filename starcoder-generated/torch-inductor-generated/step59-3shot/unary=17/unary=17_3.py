
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_1 = torch.nn.ConvTranspose2d(1, 64, kernel_size=11, stride=6, padding=2, dilation=1, output_padding=1)
    def forward(self, input_tensor):
        x = self.deconv_1(input_tensor)
        return x
# Inputs to the model
input_tensor = torch.randn(1, 1, 11, 11)
