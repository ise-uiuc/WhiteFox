
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(4, 64, 3, padding=1)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64, 128, 4, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(128, 256, 4, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v1 = torch.relu(v1)
        v2 = self.conv_transpose1(v1)
        v2 = torch.relu(v2)
        v3 = self.conv_transpose2(v2)
        v3 = torch.relu(v3)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
