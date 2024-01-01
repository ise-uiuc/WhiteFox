
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, (2, 4), 3, output_padding=(1, 2))
        self.linear = torch.nn.Linear(1344, 16)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = v6.flatten(start_dim=1)
        v8 = self.linear(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 12, 16)
