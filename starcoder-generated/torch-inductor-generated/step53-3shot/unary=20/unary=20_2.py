
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 4, kernel_size=(5, 8), stride=(4, 6), padding=(2, 3), dilation=(2, 1))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 720, 477, requires_grad=True)
