
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 12, 2, stride=4, padding=1)
        self.conv2 = torch.nn.Conv2d(12, 32, 3, stride=2, padding=1, dilation=2)
        self.layer_norm1 = torch.nn.LayerNorm((32,))
        self.conv3 = torch.nn.Conv2d(32, 64, 1)
        self.layer_norm2 = torch.nn.LayerNorm((64,), elementwise_affine=False)
        self.layer_norm3 = torch.nn.LayerNorm((64,))
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 64, 1, stride=1, padding=1)
        self.layer_norm4 = torch.nn.LayerNorm((64,))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.layer_norm1(v2)
        v4 = self.conv3(v3)
        v5 = self.layer_norm2(v4)
        v6 = v5 * 0.5
        v7 = v5 * v5 * v5
        v8 = v7 * 0.044715
        v9 = v5 + v8
        v10 = v9 * 0.7978845608028654
        v11 = torch.tanh(v10)
        v12 = v10 + 1
        v13 = v6 * v11
        v14 = v12 * v3
        v15 = self.layer_norm3(v14)
        v16 = self.conv_transpose1(v15)
        v17 = self.layer_norm4(v16)
        v18 = v17 * v12
        return v13
# Inputs to the model
x1 = torch.randn(2, 6, 7, 9)
