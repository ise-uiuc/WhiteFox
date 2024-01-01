
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, (3, 4), stride=(1, 1), padding=(3, 0), bias=False, groups=2, dilation=2, padding_mode='same')
    def forward(self, x, y):
        v1 = self.conv(x)
        v2 = v1 - y
        return v2
# Inputs to the model,
x = torch.randn(1, 2, 6, 2)
y = torch.randn(1, 4, 4, 4)
