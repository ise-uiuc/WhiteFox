
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 5, 4, stride=8, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v6 = v8 = v4 = v3 = v2 = v1
        return v8
# Inputs to the model
x1 = torch.randn(1, 32, 7, 7)
