
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = F.relu(self.conv_transpose(x1))
        v2 = v1 > 0
        v3 = F.leaky_relu(v1, negative_slope=0.1)
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
