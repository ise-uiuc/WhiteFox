
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 256, 3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 256, 3, stride=2, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        # replace the last mul(0.5*convolution, 0.7071067811865476*convolution,...) with a layer type that mimics
        v7 = torch.nn.Sigmoid()(self.conv1(v6))
        return torch.nn.ReLU()(v7)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 64, 64)
