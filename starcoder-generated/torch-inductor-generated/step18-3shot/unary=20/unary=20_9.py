
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sigmoid = torch.sigmoid
        self.conv_t = torch.nn.ConvTranspose2d(3, 7, kernel_size=(7, 7), stride=(2, 2), bias=False)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 100, 200)
