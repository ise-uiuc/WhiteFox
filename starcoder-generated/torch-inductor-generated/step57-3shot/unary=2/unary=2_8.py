
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1331, 551, kernel_size=1, stride=1, bias=True)
        self.linear_ = torch.nn.Linear(in_features=23, out_features=24, bias=True)
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
        v10 = v9.flatten(start_dim=1)
        v11 = self.linear_(v10)
        return v11
# Inputs to the model
# Note: The number of input channels must be greater than 2. The output size (height * width * depth) must be greater than 2.
x1 = torch.randn(1, 1331, 72, 104, 8)
