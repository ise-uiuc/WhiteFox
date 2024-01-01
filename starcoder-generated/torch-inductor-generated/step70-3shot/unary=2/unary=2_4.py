
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 6, 2, stride=(3, 1), padding=(1, 2), output_padding=(1, 0), groups=1)
        self.group_norm = torch.nn.GroupNorm(6, num_channels=6)
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
        v10 = self.group_norm(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 10, 10)
