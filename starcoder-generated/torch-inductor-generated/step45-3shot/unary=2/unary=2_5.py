
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(8, 26, 9)
        self.conv_transpose = torch.nn.ConvTranspose2d(26, 9, 2, stride=2, padding=3, dilation=2, output_padding=2)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.flatten(v1, start_dim=1)
        v3 = torch.reshape(v2, (x1.shape[0], 26, 5, 5))
        v4 = self.conv_transpose(v3) * 0.7978845608028654
        v5 = torch.flatten(v4, start_dim=1)
        v6 = torch.reshape(v5, (x1.shape[0], 18, 2, 4))
        return v6
# Inputs to the model
x1 = torch.randn(32, 8, 2, 4)
