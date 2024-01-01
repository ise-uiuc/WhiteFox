
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(37, 42, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(x1)
        return v1
# Inputs to the model
x1 = torch.randn(2, 37, 128, 95)
