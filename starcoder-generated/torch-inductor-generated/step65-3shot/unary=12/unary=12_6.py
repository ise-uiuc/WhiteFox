
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 16, kernel_size=2, stride=2)
        self.conv_next = torch.nn.Conv2d(16, 2, kernel_size=(3, 3), stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = F.leaky_relu(x1, negative_slope=0.20000000298023224)
        v1 = F.gelu(v1)
        v2 = v1.softmax(dim=1)
        v3 = v1.tanh()
        v4 = v1.tanh()
        v5 = self.conv(v1)
        v5 = F.sigmoid(v5)
        v5 = v5.mul(v1)
        v6 = self.conv_next(v5)
        return v5
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
