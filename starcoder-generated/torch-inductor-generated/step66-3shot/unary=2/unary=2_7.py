
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 8, 3, stride=1, padding=0, output_padding=1)
        self.avg_pool2d = torch.nn.AvgPool2d(2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.avg_pool2d(v9)
        return v10
# Inputs to the model
x1 = torch.randn(4, 8, 2, 2)
