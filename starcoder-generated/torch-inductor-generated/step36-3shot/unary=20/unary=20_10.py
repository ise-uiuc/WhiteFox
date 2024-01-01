
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=(6, 3), stride=(2, 1), dilation=(2, 1))
    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        t2 = torch.sigmoid(t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 1, 57, 38)
