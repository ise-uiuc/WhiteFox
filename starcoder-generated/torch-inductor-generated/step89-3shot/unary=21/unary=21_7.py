
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        return x
# Inputs to the model
x = torch.randn(10, 3, 224, 224)
