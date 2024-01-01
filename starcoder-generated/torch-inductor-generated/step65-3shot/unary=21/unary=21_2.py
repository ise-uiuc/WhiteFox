
class Model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2,  padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv2(x)
        x = torch.nn.functional.gelu(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        return x
kernel_size = 5
# Inputs to the model
x = torch.randn(1, 3, 299, 299)
