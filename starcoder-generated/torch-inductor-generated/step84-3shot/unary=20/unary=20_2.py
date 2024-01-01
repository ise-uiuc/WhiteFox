
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, kernel_size=(1, 5), stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = torch.sigmoid(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
