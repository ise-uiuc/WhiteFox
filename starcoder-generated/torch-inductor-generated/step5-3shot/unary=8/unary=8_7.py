
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.functional.conv_transpose2d
    def forward(self, x1):
        x1 = self.conv_transpose(input=x1, weight=0.1*torch.eye(8*9*3).reshape(8, 3, 9, 9), bias=1, stride=2, padding=1, output_padding=1, groups=1, dilation=2)
        v1 = x1 + 3
        v2 = torch.clamp(v1, min=0)
        v3 = torch.clamp(v2, max=6)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
