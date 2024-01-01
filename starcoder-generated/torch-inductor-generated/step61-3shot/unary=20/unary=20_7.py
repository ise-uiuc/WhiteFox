
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 15, kernel_size=2, stride=6, padding=1)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 119, 21)
