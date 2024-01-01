
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(3, 32, 3, padding=2, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose2(v4)
        v6 = torch.relu(v5)
        v7 = v5 - v6
        return v5 + v6
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
