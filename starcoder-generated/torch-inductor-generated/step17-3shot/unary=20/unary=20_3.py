
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(96, 14, kernel_size=(3, 3))
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = torch.sigmoid(v1)
        v3 = v2.view([1, 14, 44, 44])
        return v2
# Inputs to the model
x1 = torch.randn(1, 96, 44, 44)
