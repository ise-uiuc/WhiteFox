
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(3, 8, (15, 15))
        self.transpose_layer = torch.nn.ConvTranspose2d(3, 8, 15)
    def forward(self, x1):
        v1 = self.transpose_layer(x1)
        v2 = self.conv_layer(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
