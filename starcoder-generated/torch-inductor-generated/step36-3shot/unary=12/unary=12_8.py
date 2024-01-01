
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 33, (0,0), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False, padding_mode='zeros')
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        return v1 * v2
# Inputs to the model
x1 = torch.randn(1, 64, 1, 1)
