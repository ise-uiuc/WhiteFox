
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1)
    def forward(self, x3):
        v1 = self.conv(x3)
        v2 = v1 - 0.1
        return v2
# Inputs to the model
x3 = torch.randn(1, 1, 70, 70)
