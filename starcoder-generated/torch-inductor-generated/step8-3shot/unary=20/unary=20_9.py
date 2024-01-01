
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 4, kernel_size=(2, 1), stride=(4, 1))
    def forward(self, x1):
        x_1 = self.conv_t(x1)
        x_2 = torch.sigmoid(x_1)
        return x_2
# Inputs to the model
x = torch.randn(2, 2, 16, 2)
