
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 64, 2, stride=3, padding=4)
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.dropout(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
