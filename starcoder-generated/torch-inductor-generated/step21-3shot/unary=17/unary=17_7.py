
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(512, 256, 1)
        self.conv_transpose1 = torch.nn.ConvTranspose1d(256, 128, 1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(128, 3, 1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(4, 512, 7)
