
class Model(torch.nn.Module):
    def __init__(self, N, C, H, W):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(N, C, H, W)

    def forward(self, x):
        y = self.conv1(x)
        y = torch.cat((y, y), dim = -1)
        return y
# Inputs to the model
N, C, H, W=256, 512, 14, 14
x = torch.randn(N, C, H, W)
