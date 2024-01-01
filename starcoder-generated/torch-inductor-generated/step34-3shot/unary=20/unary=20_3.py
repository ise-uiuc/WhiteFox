
class ConvTransposeExample(nn.Module):
    def __init__(self):
        super(ConvTransposeExample, self).__init__()
        self.layer = nn.ConvTranspose2d(16, 16, kernel_size=5)

    def forward(self, x):
        x = self.layer(x)
        return x
# Inputs to the model
inp = torch.randn(1, 16, 256, 128)
net = ConvTransposeExample()
