
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, kernel_size=(3, 2), stride=(2, 1), padding=(2, 0), dilation=(2, 1), output_padding=(1, 1), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        x = self.conv_t(x)
        return x
# Inputs to the model
x = torch.randn(2, 4, 10, 10)

