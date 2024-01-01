
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=(1, 2), stride=(1, 1), padding=(0, 0), dilation=2, groups=4, padding_mode='circular')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = torch.relu(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
