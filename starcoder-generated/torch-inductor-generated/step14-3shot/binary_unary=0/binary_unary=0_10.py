
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = numpy.transpose(v1, (0, 1, 3, 2))
        v3 = numpy.reshape(v2, (v2.shape[0], v2.shape[1], v2.shape[2]*v2.shape[3]))
        x2 = v3.reshape(-1, 64, 64, 1)
        v4 = self.conv2(x2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
