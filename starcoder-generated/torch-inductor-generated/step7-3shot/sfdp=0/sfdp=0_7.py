
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Please change the following line to make this layer 1x1 conv
        # self.conv = torch.nn.Conv2d()
        self.key2 = torch.nn.Conv2d(64, 16, 1, stride=1, padding=0)
        self.value = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
 
    def forward(self, x):
        # TODO: Please change the following line to use 1x1 conv as query
        # v1 = self.conv(x)
        k2 = self.key2(x)
        # TODO: Please change the following line to use 1x1 conv as key
        # k1 = self.conv(x)
        v = self.value(x)
        return torch.matmul(k2, v.transpose(-2, -1)) / math.sqrt(64)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64, 1, 1)
