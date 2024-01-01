
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 12, kernel_size=(3, 6), stride=(3, 1), padding=(2, 1), output_padding=(1, 2))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 19, 72, 256, requires_grad=True)
