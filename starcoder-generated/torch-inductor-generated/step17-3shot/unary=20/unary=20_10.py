
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, kernel_size=2, stride=2, padding=1)
    def forward(self, x1):
        t1 = self.conv_transpose(x1)
        t2 = torch.sigmoid(t1)
        return t2
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
