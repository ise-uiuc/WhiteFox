
class Model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, kernel_size)
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = t1 > 1
        t3 = t1 * 6.732
        t4 = torch.where(t2, t1, t3)
        return t4
kernel_size = (2, 4)
# Inputs to the model
x = torch.randn(1, 1, 16, 25)
