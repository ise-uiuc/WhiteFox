
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(3, 3, 2, stride=1, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = F.leaky_relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
