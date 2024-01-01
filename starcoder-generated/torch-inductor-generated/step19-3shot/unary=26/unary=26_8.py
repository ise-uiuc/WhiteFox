
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 2, 4, 2)
        self.batch_norm = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = self.batch_norm(x2)
        return x3
# Inputs to the model
x1 = torch.randn(25, 4, 28, 28)
