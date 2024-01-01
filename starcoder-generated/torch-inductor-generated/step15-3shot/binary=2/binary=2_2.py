
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(64, 32, kernel_size=(256), stride=(1), padding=(32))
    def forward(self, x):
        v = self.conv(x)
        return v - 0.125
# Inputs to the model
x = torch.randn(1, 64, 256, 256)
