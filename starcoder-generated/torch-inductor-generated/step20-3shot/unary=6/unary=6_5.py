
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(1, 7), stride=(1, 1), padding=(1, 3), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv2d(64, 4, kernel_size=(7, 1), stride=(1, 1), padding=(3, 1), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 256)
