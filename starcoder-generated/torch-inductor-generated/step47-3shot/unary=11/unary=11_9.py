
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = nn.functional.pixel_shuffle(x1, 4)
        v2 = torch.reshape(v1, (1, 32, 32, 32))
        v3 = torch.transpose(v2, 2, 3)
        v4 = torch.transpose(v3, 1, 2)
        v5 = nn.functional.conv_transpose3d(v4, weight=None, bias=None, stride=(1, 1, 4), padding=(0, 0, 1), output_padding=(0, 0, 0), groups=1, dilation=1)
        v6 = v5 + 3
        v7 = v6 + v1
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
