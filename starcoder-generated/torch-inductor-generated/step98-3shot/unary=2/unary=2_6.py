
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.conv_transpose3d(x1, torch.zeros(1, 1, 2, 2, 4), bias=torch.zeros(1), stride=1, padding=0, output_padding=0, groups=1, dilation=1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 2, 4, 10, 10)
