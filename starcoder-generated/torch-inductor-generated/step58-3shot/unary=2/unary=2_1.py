
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 7, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = torch.tanh(v2)
        v4 = v3 + 1
        v5 = v2 * v4
        return v5
# Inputs to the model
x1 = torch.rand(2, 1, 11, 18)
