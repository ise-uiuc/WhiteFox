
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose_conv1x1 = torch.nn.ConvTranspose2d(1, 1, 1, stride=1)

    def forward(self, x1):
        x = self.transpose_conv1x1(x1)
        x = x.reshape(x.shape[0], -1)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 9, 9)
