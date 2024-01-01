
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 15, kernel_size=(2, 2), stride=(2, 2), bias=False)
    def forward(self, xn):
        v0 = torch.flatten(xn, 1)
        v1 = torch.reshape(v0, (-1, 15, 9, 9))
        v2 = self.conv_t(v1)
        v3 = torch.permute(v2, [0, 2, 3, 1])
        v4 = torch.sigmoid(v3)
        v5 = torch.reshape(v4, (-1, 125))
        return v5
# Inputs to the model
xn = torch.randn(3, 15)
