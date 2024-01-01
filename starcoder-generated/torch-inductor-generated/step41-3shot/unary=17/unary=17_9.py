
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(5, 5, 3, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 8, 3, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 8, 3, padding=3)
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = torch.relu(v2)
        v4 = self.conv_transpose(v3)
        v5 = torch.relu(v4)
        v6 = self.conv_transpose1(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 5, 128, 128)
