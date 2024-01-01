
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Create a max pooling layer with kernel size 3 and stride 2
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=1, padding=2)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.maxpool1(v1)
        v3 = self.conv1(v2)
        v4 = self.maxpool1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 85, 85)
