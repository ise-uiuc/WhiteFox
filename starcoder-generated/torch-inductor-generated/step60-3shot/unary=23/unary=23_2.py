
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_no_groups = torch.nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose_no_groups(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 256, 42, 53)
