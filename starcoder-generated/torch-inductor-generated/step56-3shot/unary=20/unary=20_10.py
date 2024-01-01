
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(20, 20, kernel_size=(1, 1), bias=False)
        self.conv_t2 = torch.nn.ConvTranspose3d(20, 20, kernel_size=(2, 1, 1), bias=True)
        self.conv_t3 = torch.nn.ConvTranspose3d(20, 1, kernel_size=(2, 2, 2), bias=True)
    def forward(self, x):
        v1 = self.conv_t1(x)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_t2(v2)
        v4 = self.conv_t3(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 20, 32, 32)
