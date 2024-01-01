
class Model(torch.nn.Module):
    def __init__(self, kernel_size=(2, 3), padding=(8, 11), stride=(1, 2)):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size, padding=padding, stride=stride)
    def forward(self, x):
        return self.conv(x)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
