
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - torch.rand(1, 32, 128, 128)
        return v2
# Inputs to the model
x1 = torch.rand(1, 32, 64, 64)
