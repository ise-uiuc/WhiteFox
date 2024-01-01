
class Model(torch.nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(9216*4, 128)
        self.relu = torch.nn.ReLU()

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.reshape((v1.shape[0], -1))
        v3 = self.linear(v2)
        v4 = self.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
