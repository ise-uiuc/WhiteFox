
class MyModule(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=13, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 231, 456)
