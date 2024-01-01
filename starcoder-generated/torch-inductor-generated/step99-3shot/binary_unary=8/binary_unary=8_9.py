
class Model(torch.nn.Module):
    def __init__(self):
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(32, 3, 512, 512)
