
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(4, 32, 8, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 4, 8, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose(v2)
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 16, 16)
