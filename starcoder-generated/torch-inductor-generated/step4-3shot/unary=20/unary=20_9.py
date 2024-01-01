
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.conv_transpose2d(input=x1, weight=x1, bias=None, stride=2, padding=1, output_padding=0, groups=1, dilation=1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
