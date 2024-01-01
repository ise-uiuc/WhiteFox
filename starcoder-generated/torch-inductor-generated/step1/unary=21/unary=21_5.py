
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(16, 6, 1)
        self.conv1 = torch.nn.Conv2d(16, 6, 1)

    def forward(self, x0):
        v0 = self.conv0(x0)
        v1 = self.conv1(x0)
        v4 = v0.tanh()
        v5 = v4 * v1
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 20, 20)
