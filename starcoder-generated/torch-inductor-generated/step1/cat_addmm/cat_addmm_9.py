
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return torch.cat([-1 * v1, v2], axis=-1)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
