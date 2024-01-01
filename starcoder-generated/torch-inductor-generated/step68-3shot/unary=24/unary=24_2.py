
class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), dilation=(0, 0))

    def forward(self, x):
        negative_slope = 0.85498
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 5, 7)
