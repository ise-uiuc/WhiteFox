
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.relu(x)
        return self.flatten(x2)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
