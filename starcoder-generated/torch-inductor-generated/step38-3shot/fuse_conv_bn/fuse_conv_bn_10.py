
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # Since x is used by the output, we cannot fuse it
        return x, x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
