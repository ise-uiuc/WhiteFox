
class ReshapeWrapper(torch.nn.Module):
    def __init__(self, reshape_size):
        super().__init__()
        self.reshape_size = reshape_size

    def forward(self, x):
        x = x.reshape(self.reshape_size)
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.block1 = ReshapeWrapper((56, 16, 768))
 
    def forward(self, x):
        v1 = self.conv(x)
        v3 = self.block1(v1)
        v5 = torch.sum(v3, -1)
        v6 = torch.sum(v5, -1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model with the corresponding shape
x = torch.randn(1, 3, 64, 64)
m.block1.reshape_size = x.shape
