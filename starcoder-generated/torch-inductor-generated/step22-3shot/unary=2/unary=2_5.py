
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 5, stride=2, padding=3, output_padding=3, dilation=3, groups=2, bias=True)
    def forward(self, x1):
        v1 = np.random.randn(3, 5, 5, 7)
        v2 = torch.from_numpy(v1)
        v2.requires_grad = True
        v2.retain_grad()
        self.conv_transpose.weight = torch.nn.Parameter(v2)
        self.conv_transpose.bias = torch.nn.Parameter(torch.randn(v2.size(0)))
        v3 = self.conv_transpose(x1)
        v4 = v3 * 0.5
        v5 = v3 * v3 * v3
        v6 = v5 * 0.044715
        v7 = v3 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v4 * v10
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 15, 15)
