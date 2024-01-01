
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, kernel_size=(3, 5), stride=(4, 4), padding=0, dilation=(0, 0))
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(1, v1.shape[1], v1.shape[2] + 1, v1.shape[3] + 1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
