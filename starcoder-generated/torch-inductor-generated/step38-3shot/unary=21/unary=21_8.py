
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=5, stride=3, padding=2)
    def forward(self, x):
        x = torch.tanh(x)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(10, 3, 75, 75)
