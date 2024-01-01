
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv_args = {
            'in_channels': 31,
            'out_channels': 14,
            'kernel_size': 5,
          'stride': 12,
            'padding': 1
        }
        self.conv1 = torch.nn.Conv2d(**conv_args)
        self.conv2 = torch.nn.Conv2d(**conv_args)
    def forward(self, x421):
        v1 = self.conv1(x421)
        v2 = v1 * 0.5
        v3 = v1 * v1
        v4 = v3 * v1
        v5 = v4 * 0.044715
        v6 = v1 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v2 * v9
        v11 = self.conv2(v10)
        v12 = v11 * 0.5
        v13 = v11 * v11
        v14 = v13 * v11
        v15 = v14 * 0.044715
        v16 = v11 + v15
        v17 = v16 * 0.7978845608028654
        v18 = torch.tanh(v17)
        v19 = v18 + 1
        v20 = v12 * v19
        return v20
# Inputs to the model
x421 = torch.randn(1, 31, 43, 25)
