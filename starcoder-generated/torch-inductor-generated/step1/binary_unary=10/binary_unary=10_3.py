
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(50, 50, 3, stride=1, padding=1)

    def forward(self, x):
        y = self.conv(x)
        z = torch.relu(y, other="fused_add_relu")
        return z

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 50, 64, 64)
