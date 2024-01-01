
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv_transpose_relu = torch.nn.ConvTranspose2d(3, 3, 3, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose_relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
