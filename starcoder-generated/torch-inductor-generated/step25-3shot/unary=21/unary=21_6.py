
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 15, stride=2)
        self.conv2 = torch.nn.Conv2d(1, 1, 8, stride=3, padding=15)
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(5, 5, 5, 5)
