
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1  = torch.nn.Conv2d(64,  16, (3,3))
        self.conv2  = torch.nn.Conv2d(16,  32, (5,5), 1, 1, 1)
        self.conv3  = torch.nn.Conv2d(32,  64, (1,3), 1, 1, 0)
        for i in range(2):
            self.add_module(str(i), ConvBn2d(64, 64))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        for i in range(2):
            v3 += getattr(self, str(i))(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 32, 64)
