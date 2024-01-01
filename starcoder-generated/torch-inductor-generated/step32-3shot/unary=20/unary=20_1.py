
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_2 = torch.nn.ConvTranspose2d(8, 7, kernel_size=7, stride=(5, 4), groups=2, dilation=(7, 1))
    def forward(self, x1):
        v1 = self.conv_t_2(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 8, 777, 777)
