
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 1, 1, stride=2, padding=7, output_padding=7)
    def forward(self, x1):
        v1 = v2 = self.conv_transpose(x1)
        v3 = v2 * 0.125
        v5 = torch.nn.functional.one_hot(v2, num_classes=12)
        v7 = torch.nn.functional.dropout(v5, p=0.0, training=True, inplace=False)
        v6 = v3 * v7
        return v6
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
