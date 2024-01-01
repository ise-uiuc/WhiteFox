
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, kernel_size=(1, 1), stride=(2, 3), dilation=(3, 4))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.7
        return v2
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
