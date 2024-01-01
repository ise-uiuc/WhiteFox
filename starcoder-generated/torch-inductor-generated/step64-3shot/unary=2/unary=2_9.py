
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(4, 3, (3, 3, 1), stride=(1, 1, 1), padding=(13, 10, 11), output_padding=(0, 0, 1), groups=4, dilation=(2, 2, 2), pads_count=15, weight_numel=342, bias_false=False)
        self.conv = torch.nn.Conv3d(4, 3, (3, 3, 1), stride=(3, 2, 2), padding=(1, 1, 2), dilation=(2, 2, 2))
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
        v10 = self.conv(x1)
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 100, 21, 1)
