
class Model():
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
torch.manual_seed(0)
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
