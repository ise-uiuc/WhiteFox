
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(4, 4, kernel_size=4, stride=4)
        self.conv_t2 = torch.nn.ConvTranspose2d(3, 6, kernel_size=3, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose2d(1, 2, kernel_size=3, stride=1)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v1)
        v4 = torch.sigmoid(v2)
        return v4
# Inputs to the model
x1 = torch.randn(2, 4, 7, 5)
