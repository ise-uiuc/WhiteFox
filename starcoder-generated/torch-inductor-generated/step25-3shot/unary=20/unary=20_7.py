
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1536, 49, kernel_size=(6, 6), stride=(2, 2))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v1
# Inputs to the model
x1 = torch.randn(10, 1536, 13, 13)
