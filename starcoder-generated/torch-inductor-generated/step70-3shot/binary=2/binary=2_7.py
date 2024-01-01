
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, (2,32), stride=(2, 2))
    def forward(self, x):
        y = self.conv(x)
        y = y - 123
        return y
# Inputs to the model
x = torch.randn(1, 3, 128, 256)
