
class Model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 299, 299)
