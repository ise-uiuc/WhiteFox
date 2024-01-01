
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(2, 2), stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1, bias=True, dilation=1)
    def forward(self, x1, x2):
        v1 = torch.add(x1, 1, alpha=1)
        v2 = x2 + 1.0
        v3 = torch.nn.functional.adaptive_avg_pool2d(v2, (1, 1))
        v4 = torch.nn.functional.hardtanh(v3, 0.0, 6.0)
        v6 = torch.add(v4, 1, alpha=1)
        v7 = torch.cat((v1, v6), 1)
        v8 = torch.flatten(v7, start_dim=1)
        v10 = torch.nn.functional.linear(v8, 10, bias=True)
        v12 = v10 + 1.0
        v14 = torch.sigmoid(v12)
        v16 = v16 + 1.0
        v18 = torch.sigmoid(v16)
        return v18
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 2, 2, 2)
